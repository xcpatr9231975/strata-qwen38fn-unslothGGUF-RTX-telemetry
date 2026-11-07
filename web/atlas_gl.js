/* ============================================================================
 * atlas_gl.js — WebGL2 renderer for the Qwen3.8-Flash-Next physical weight
 * atlas + live Strata inference telemetry.
 *
 * WHAT IS DRAWN (all geometry derived from window.ATLAS_META / ATLAS_HEAT,
 * produced by tools/build_atlas.py — nothing is hard-coded except layout):
 *
 *  EMBEDDING  : token_embd.weight slab, vocab=248320 x hidden=2560.
 *               512x64 sampled row grid standing vertically; a "lookup
 *               position" bar sweeps rows on every emitted token, i.e. the
 *               embedding *position* (row index space) of the token being
 *               fed into the forward pass.
 *  48 LAYER PLATES, laid out 8 columns x 6 rows. Each plate:
 *               - RIM colour: cyan = full-attention block (L%4==3),
 *                 magenta = SSM block, plus gold notch when PLE attaches.
 *               - 32x16 = 512 expert cells (one per routed expert of that
 *                 MoE layer). Cell base colour = profile heat (co-routing
 *                 edge count from expert-profile.bin, viridis-like ramp).
 *                 Cells with heat>0 are in the warmed resident cache
 *                 (engine: "pre-filled 5175 of 5175 slots from the
 *                 profile ... no eviction"). Amber flash = routed active
 *                 for the current token (top-10-of-512 per layer; the
 *                 engine does not expose per-token expert ids over the API,
 *                 so activation is *synthesised* heat-weighted and the HUD
 *                 flags it "synthetic routing").
 *               - ROUTER BAR: ffn_gate_inp.weight (2560x512 bf16) strip.
 *  HEAD       : output.weight slab (2560x248320, transposed embedding).
 *  MTP        : draft-layer slab (mtp/rt pack: 29 dense tensors + 512
 *               experts, 1382400 B/expert). Green spec-particles loop
 *               head -> mtp -> verify at drafts_offered/accepted rates.
 *  TOKEN PARTICLES travel the polyline embed -> blk0..47 -> head at the
 *               live decode speed (tok_s); prefill emits a fast burst.
 *
 * INTERACTION: drag = orbit, wheel = zoom, right-drag = pan, click = pick
 * expert/plate (byte-located via native_experts.txt regions), D = toggle
 * dark grid, R = reset camera.
 * ========================================================================== */
"use strict";

/* ------------------------------ math helpers ----------------------------- */
const V3 = (x=0,y=0,z=0)=>[x,y,z];
const sub=(a,b)=>[a[0]-b[0],a[1]-b[1],a[2]-b[2]];
const add=(a,b)=>[a[0]+b[0],a[1]+b[1],a[2]+b[2]];
const scale=(a,s)=>[a[0]*s,a[1]*s,a[2]*s];
const dot=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const norm=a=>{const l=Math.hypot(...a)||1;return [a[0]/l,a[1]/l,a[2]/l];};
const M4={
  ident:()=>new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]),
  persp:(fovy,aspect,near,far)=>{const f=1/Math.tan(fovy/2),nf=1/(near-far);
    return new Float32Array([f/aspect,0,0,0, 0,f,0,0, 0,0,(far+near)*nf,-1, 0,0,2*far*near*nf,0]);},
  look:(eye,center,up)=>{const z=norm(sub(eye,center)),x=norm(cross(up,z)),y=cross(z,x);
    return new Float32Array([x[0],y[0],z[0],0, x[1],y[1],z[1],0, x[2],y[2],z[2],0,
      -dot(x,eye),-dot(y,eye),-dot(z,eye),1]);},
  mul:(a,b)=>{const o=new Float32Array(16);
    for(let i=0;i<4;i++)for(let j=0;j<4;j++){let s=0;
      for(let k=0;k<4;k++)s+=a[k*4+j]*b[i*4+k]; o[i*4+j]=s;} return o;}
};

/* colour ramps ------------------------------------------------------------- */
function heatColor(h){ /* 0..inf -> dark navy -> teal -> gold (viridis-ish) */
  const t=Math.min(1,Math.log1p(h)/Math.log1p(8));
  const r=0.04+0.92*t*t, g=0.10+0.72*t, b=0.30-0.22*t;
  return [r,g,b];
}
const KIND_COL={full_attention:[0.10,0.85,0.95], ssm:[0.85,0.25,0.90]};

/* --------------------------- layout constants ---------------------------- */
const COLS=8, ROWS=6, PLATE_W=40, PLATE_H=26, GAP_X=52, GAP_Z=40;
const GRID_C=32, GRID_R=16, CELL=1.02, CELL_GAP=0.135; /* 32x16=512 experts */
const plateCenter=L=>[ (L%COLS-(COLS-1)/2)*GAP_X, 0, Math.floor(L/COLS)*GAP_Z ];

/* ------------------------------ GL bootstrap ------------------------------ */
class AtlasGL{
  constructor(canvas,meta,heat){
    this.cv=canvas; this.meta=meta; this.heat=heat;
    this.gl=canvas.getContext("webgl2",{antialias:true,alpha:false});
    if(!this.gl) throw new Error("WebGL2 unavailable");
    const gl=this.gl;
    this.prog=this._makeProg(AtlasGL.VS_QUAD, AtlasGL.FS_QUAD);
    this.aPos=gl.getAttribLocation(this.prog,"aPos");
    this.aOff=gl.getAttribLocation(this.prog,"aOff");
    this.aSz =gl.getAttribLocation(this.prog,"aSz");
    this.aCol=gl.getAttribLocation(this.prog,"aCol");
    this.aKind=gl.getAttribLocation(this.prog,"aKind");
    this.uVP=gl.getUniformLocation(this.prog,"uVP");
    this.uY=gl.getUniformLocation(this.prog,"uY");
    /* quad corners (CCW triangle strip) */
    this.cornerBuf=gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER,this.cornerBuf);
    gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([-1,-1, 1,-1, -1,1, 1,1]),gl.STATIC_DRAW);
    /* camera */
    this.orbAz=0.0; this.orbEl=0.72; this.orbDist=460; this.target=V3(0,0,GAP_Z*2.7);
    this.fly=null;
    this._resize(); this._buildScene();
    /* per-frame mutable state */
    this.flash=new Float32Array(meta.dims.layers*meta.dims.experts); /* decay 0..1 */
    this.layerAct=new Float32Array(meta.dims.layers); /* forward-pass arrival glow */
    this.fwdStage={name:"idle",label:null,layer:-1};   /* (a) forward-pass tracker */
    this.particles=[]; this.tokAcc=0; this.specAcc=0;
    this.telemetry={}; this.pick=null;
    gl.enable(gl.DEPTH_TEST); gl.clearColor(0.015,0.018,0.03,1);
  }

  /* ------------------------------- shaders ------------------------------- */
  static VS_QUAD=`#version 300 es
  in vec2 aPos;      /* unit quad corner */
  in vec2 aOff;     /* instance centre in world XZ */
  in vec2 aSz;      /* instance half-size in X,Z */
  in vec3 aCol;     /* instance colour */
  in float aKind;   /* 0 plate rim,1 plate,2 expert cell,3 bar,4 particle,5 slab,6 grid */
  uniform mat4 uVP; uniform float uY;
  out vec3 vCol; out float vKind; out vec2 vUV;
  void main(){
    vec3 w=vec3(aOff.x+aPos.x*aSz.x, uY + (aKind==4.0? 2.2:0.0), aOff.y+aPos.y*aSz.y);
    gl_Position=uVP*vec4(w,1.0);
    vCol=aCol; vKind=aKind; vUV=aPos;
  }`;
  static FS_QUAD=`#version 300 es
  precision mediump float;
  in vec3 vCol; in float vKind; in vec2 vUV;
  out vec4 o;
  void main(){
    float d=max(abs(vUV.x),abs(vUV.y));       /* square SDF for rimming */
    float edge=smoothstep(0.86,0.94,d);       /* 1 near border */
    if(vKind==2.0){                            /* expert cell */
      float lit=mix(0.82,1.35,edge);           /* brighter rim */
      o=vec4(min(vCol*lit,1.0),1.0);
    } else if(vKind==4.0){                     /* particle: soft disc */
      float a=smoothstep(1.0,0.35,length(vUV));
      o=vec4(vCol*a,a);
    } else {
      o=vec4(vCol + edge*0.06,1.0);
    }
  }`;

  _makeProg(vsSrc,fsSrc){
    const gl=this.gl;
    const comp=(t,src)=>{const s=gl.createShader(t);gl.shaderSource(s,src);gl.compileShader(s);
      if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(s));return s;};
    const p=gl.createProgram();
    gl.attachShader(p,comp(gl.VERTEX_SHADER,vsSrc));
    gl.attachShader(p,comp(gl.FRAGMENT_SHADER,fsSrc));
    gl.linkProgram(p);
    if(!gl.getProgramParameter(p,gl.LINK_STATUS))throw new Error(gl.getProgramInfoLog(p));
    return p;
  }

  /* ------------------------------ scene data ------------------------------ */
  /* instance arrays are (off.x,off.y,sz.x,sz.y,col[3],kind) tuples */
  _push(arr,off,sz,col,kind){arr.push(off[0],off[1],sz[0],sz[1],col[0],col[1],col[2],kind);}

  _buildScene(){
    const gl=this.gl, meta=this.meta, dims=meta.dims;
    const stat=[], dyn=[];   /* static instances / expert cells (dynamic) */
    /* per-layer plates */
    this.layers=[];
    for(let L=0;L<dims.layers;L++){
      const c=plateCenter(L);
      const kindC=KIND_COL[meta.layers[L].kind]||[0.5,0.5,0.5];
      /* rim (slightly bigger plate under the main plate, same y, drawn first) */
      this._push(stat,[c[0],c[2]],[PLATE_W/2+0.6,PLATE_H/2+0.6],scale(kindC,0.30),0);
      /* main plate */
      this._push(stat,[c[0],c[2]],[PLATE_W/2,PLATE_H/2],[0.07,0.08,0.11],1);
      /* router bar along front edge (ffn_gate_inp) */
      this._push(stat,[c[0],c[2]+PLATE_H/2-1.1],[PLATE_W/2-0.5,0.55],
                 [kindC[0]*0.4+0.25,kindC[1]*0.4+0.25,kindC[2]*0.4+0.25],3);
      /* PLE notch (blk.1 hosts ple_* tensors) */
      if(meta.layers[L].ple) this._push(stat,[c[0]-PLATE_W/2+2,c[2]-PLATE_H/2+1.2],[1.6,1.6],[1,0.8,0.15],1);
      /* expert cell base geometry (colours re-uploaded each frame) */
      const base=[c[0]-((GRID_C-1)*(CELL+CELL_GAP))/2, c[2]+1.9-((GRID_R-1)*(CELL+CELL_GAP))/2];
      for(let r=0;r<GRID_R;r++)for(let cc=0;cc<GRID_C;cc++){
        const e=r*GRID_C+cc;
        this._push(dyn,[base[0]+cc*(CELL+CELL_GAP),base[1]+r*(CELL+CELL_GAP)],[CELL/2,CELL/2],heatColor(this.heat[L][e]),2);
      }
      this.layers.push({i:L,c});
    }
    /* embedding / head / MTP slabs (tall: drawn as rim+plate, y fixed) */
    this.embedC=[-COLS*GAP_X/2-26, 0, GAP_Z*2.5];
    this.headC  =[ COLS*GAP_X/2+26, 0, GAP_Z*2.5-6];
    this.mtpC   =[ COLS*GAP_X/2+26, 0, GAP_Z*2.5+16];
    for(const [c,col,label] of [[this.embedC,[0.35,0.75,0.35],"EMBED"],
                                [this.headC,[0.75,0.55,0.85],"HEAD"],
                                [this.mtpC,[0.25,0.90,0.35],"MTP"]]){
      this._push(stat,c,[20,14],scale(col,0.28),0);
      this._push(stat,c,[19,13],[0.08,0.10,0.09],5);
    }
    this.statBuf=this._upload(stat); this.statN=stat.length/8;
    this.dynBuf=this._upload(dyn);   this.dynN=dyn.length/8;
    /* pipeline polyline for particles */
    this.path=[...this.embedC]; for(let L=0;L<dims.layers;L++){const c=plateCenter(L);this.path.push([c[0],2,c[2]]);}
    this.path.push([...this.headC]); this.pathLen=this._pathLen();
    /* (c) KV-cache growth bar: 32 ground cells at the front-left; each cell =
       capacity/32 tokens. Green = VRAM-resident (<= engine.kv_resident),
       blue-grey = streamed from RAM, brightest = newest written cell. */
    this.kvC=[-COLS*GAP_X/2-34, 0, GAP_Z*0.45];
    this.kvDyn=[];
    for(let i=0;i<32;i++) this._push(this.kvDyn,[this.kvC[0], this.kvC[2]+i*(4.9+0.35)],[5.0,2.45],[0,0,0],3);
    this.kvBuf=this._upload(this.kvDyn); this.kvN=32;
  }

  _upload(arr){const gl=this.gl,b=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,b);
    gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(arr),gl.STATIC_DRAW);return b;}
  _pathLen(){let s=0;for(let i=1;i<this.path.length;i++)s+=Math.hypot(...sub(this.path[i],this.path[i-1]));return s;}
  pathPoint(s){ /* s in 0..1 along polyline */
    let d=s*this.pathLen;
    for(let i=1;i<this.path.length;i++){
      const p0=this.path[i-1],p1=this.path[i],l=Math.hypot(...sub(p1,p0));
      if(d<=l){const t=d/(l||1);return add(p0,scale(sub(p1,p0),t));}
      d-=l;
    }
    return this.path[this.path.length-1];
  }

  /* ------------------------------ resize ------------------------------ */
  _resize(){
    const gl=this.gl,cv=this.cv;
    const dpr=Math.min(2,window.devicePixelRatio||1);
    const w=Math.floor(cv.clientWidth*dpr),h=Math.floor(cv.clientHeight*dpr);
    if(cv.width!==w||cv.height!==h){cv.width=w;cv.height=h;}
    gl.viewport(0,0,w,h);
  }

  /* --------------------------- live telemetry --------------------------- */
  setTelemetry(t){
    this.telemetry=t;
    const live=t.live||{}, eng=t.engine||{}, kv=t.kv||{};
    this.decodeSpeed=(live.tok_s||0);
    this.phase=live.phase||"idle";
    this.prefillFrac=(live.phase==="prefill" && live.prompt_total)? (live.prompt_read||0)/live.prompt_total : null;
    this.kvCap=kv.capacity_tokens||null; this.kvSeq=kv.seq_tokens||null; this.kvResident=kv.resident_tokens||null;
  }

  /* per-frame: synthesise top-K=10 routing per layer at token ticks,
     decay flashes. The Strata API exposes only aggregate expert-cache hit
     %; per-token ids are internal, hence synthetic (labelled in HUD). */
  _stepRouting(dt){
    const dims=this.meta.dims, K=dims.top_k_experts;
    this.tokAcc+=(this.decodeSpeed||this.phase==="prefill"?Math.max(this.decodeSpeed,6):0)*dt;
    while(this.tokAcc>=1){
      this.tokAcc-=1;
      for(let L=0;L<dims.layers;L++){
        const h=this.heat[L]; const base=L*dims.experts;
        for(let k=0;k<K;k++){ /* K distinct heat-weighted picks */
          const e=Math.min(dims.experts-1,
            Math.floor(Math.pow(Math.random(),1.6)*dims.experts)); /* bias to low ids is fine: heat array is the real signal for colours */
          this.flash[base+e]=1.0;
        }
      }
      this.particles.push({s:0,life:1,col:[0.9,0.95,0.6],li:-1}); /* ONE token particle per token */
    }
    for(let i=0;i<this.flash.length;i++)if(this.flash[i]>0)this.flash[i]=Math.max(0,this.flash[i]-dt*2.2);
    /* MTP spec particles: offered rate from totals or live */
    const tot=this.telemetry.totals||{}, live=this.telemetry.live||{};
    const offered=(tot.drafts_offered||0)>0?1:0;
    if(offered) { this.specAcc+=dt*Math.min(4,this.decodeSpeed/9);
      while(this.specAcc>=1){this.specAcc-=1;this.particles.push({s:0.985,mtp:true,life:1,col:[0.3,1,0.45]});}}
    /* integrate particles; (a) light each plate the token head arrives at */
    const speed=0.14*(1+(this.decodeSpeed||0)/24);
    const dims2=this.meta.dims;
    for(const p of this.particles){
      p.s+=speed*(p.mtp?2:1)/this.meta.dims.layers* 4.6 *dt*10; if(p.s>1.05)p.life=0;
      if(!p.mtp){
        const li=Math.min(dims2.layers-1, Math.floor(p.s*(dims2.layers+1))-1);
        if(li>=0 && li!==p.li){ this.layerAct[li]=1; p.li=li; }
      }
    }
    for(let i=0;i<this.layerAct.length;i++)if(this.layerAct[i]>0)this.layerAct[i]=Math.max(0,this.layerAct[i]-dt*1.6);
    this._updateFwdStage(dims2);
    this.particles=this.particles.filter(p=>p.life>0).slice(-700);
  }

  /* (a) forward-pass stage: follow the token particle furthest along the
     polyline embed -> blk0..47 -> head; each polyline segment is sub-phase-
     bucketed into the block's internal pipeline. HUD strip + plate glow use it. */
  static FWD_STAGES=["EMB","HC0 mix","ATTN/SSM core","HC1 combine","ROUTER softmax","10/512 experts","SHARED expert","HEAD"];
  _updateFwdStage(dims){
    let best=null;
    for(const p of this.particles) if(!p.mtp && (!best||p.s>best.s)) best=p;
    if(!best){this.fwdStage={name:"idle",label:null,layer:-1};return;}
    const segs=this.path.length-1;            /* 49: embed..48 plates..head */
    const f=best.s*segs;
    let name,layer=-1;
    if(f<1) name="EMB";
    else if(f>=segs-1) name="HEAD";
    else{ layer=Math.floor(f)-1;
      const u=f-Math.floor(f);                /* 0..1 inside one block */
      name=AtlasGL.FWD_STAGES[1+Math.min(6,Math.floor(u*7))]; }
    this.fwdStage={name,label:(this.prefillFrac!=null?`prefill ${(this.prefillFrac*100).toFixed(0)}%`:this.phase),layer};
  }

  /* ------------------------------- frame ------------------------------- */
  frame(dt){
    this._resize(); this._stepRouting(dt);
    const gl=this.gl;
    const eye=add(this.target,[Math.sin(this.orbAz)*Math.cos(this.orbEl)*this.orbDist,
                              Math.sin(this.orbEl)*this.orbDist,
                              Math.cos(this.orbAz)*Math.cos(this.orbEl)*this.orbDist]);
    const vp=M4.mul(M4.persp(0.85,this.cv.width/this.cv.height,1,2600),M4.look(eye,this.target,[0,1,0]));
    this.lastVP=vp; this.lastEye=eye;
    gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
    gl.useProgram(this.prog); gl.uniformMatrix4fv(this.uVP,false,vp);

    /* draw instance batches (plates/bars/slabs share one program) */
    const drawBatch=(buf,n,y)=>{
      gl.uniform1f(this.uY,y);
      gl.bindBuffer(gl.ARRAY_BUFFER,this.cornerBuf);
      gl.enableVertexAttribArray(this.aPos); gl.vertexAttribPointer(this.aPos,2,gl.FLOAT,false,0,0); gl.vertexAttribDivisor(this.aPos,0);
      gl.bindBuffer(gl.ARRAY_BUFFER,buf);
      const S=32; /* stride bytes */
      gl.enableVertexAttribArray(this.aOff); gl.vertexAttribPointer(this.aOff,2,gl.FLOAT,false,S,0); gl.vertexAttribDivisor(this.aOff,1);
      gl.enableVertexAttribArray(this.aSz); gl.vertexAttribPointer(this.aSz,2,gl.FLOAT,false,S,8); gl.vertexAttribDivisor(this.aSz,1);
      gl.enableVertexAttribArray(this.aCol); gl.vertexAttribPointer(this.aCol,3,gl.FLOAT,false,S,16); gl.vertexAttribDivisor(this.aCol,1);
      gl.enableVertexAttribArray(this.aKind); gl.vertexAttribPointer(this.aKind,1,gl.FLOAT,false,S,28); gl.vertexAttribDivisor(this.aKind,1);
      gl.drawArraysInstanced(gl.TRIANGLE_STRIP,0,4,n);
    };
    /* expert cells: re-colour every frame (heat + amber flash).
       _dynArr rows are stride-8 (off.xy, sz.xy, col.rgb, kind); offsets/sizes/
       kind are written once by _patchDynStatic, colours every frame. */
    if(!this._dynArr){this._dynArr=new Float32Array(this.dynN*8); this._patchDynStatic();}
    const dims=this.meta.dims;
    for(let L=0;L<dims.layers;L++)for(let e=0;e<dims.experts;e++){
      const inst=L*dims.experts+e, o=inst*8, f=this.flash[L*dims.experts+e];
      const hc=heatColor(this.heat[L][e]);
      const fr=Math.min(1,f);
      this._dynArr[o+4]=hc[0]*(1-fr)+1.0*fr; this._dynArr[o+5]=hc[1]*(1-fr)+0.72*fr; this._dynArr[o+6]=hc[2]*(1-fr)+0.05*fr;
    }
    gl.bindBuffer(gl.ARRAY_BUFFER,this.dynBuf);
    gl.bufferSubData(gl.ARRAY_BUFFER,0,this._dynArr);
    drawBatch(this.statBuf,this.statN,0.02);
    drawBatch(this.dynBuf,this.dynN,0.34);
    /* (c) KV growth bar: colours re-computed per frame from telemetry */
    if(!this._kvArr){this._kvArr=new Float32Array(this.kvN*8); this._kvArr.set(this.kvDyn);}
    {
      const cap=this.kvCap||65536, res=this.kvResident||0, seq=this.kvSeq||0, per=cap/32;
      for(let i=0;i<32;i++){
        const o=i*8, lo=i*per, lit=seq>lo;
        let r,g,b;
        if(!lit){r=0.05;g=0.06;b=0.09;}
        else if(lo<res){r=0.10;g=0.80;b=0.35;}   /* VRAM-resident KV */
        else {r=0.25;g=0.45;b=0.85;}             /* overflow: streamed from RAM */
        const k=Math.abs(i-Math.min(31,Math.floor(seq/per)))<0.6?1.9:1; /* newest written cell pops */
        this._kvArr[o+4]=Math.min(1,r*k); this._kvArr[o+5]=Math.min(1,g*k); this._kvArr[o+6]=Math.min(1,b*k);
      }
      gl.bindBuffer(gl.ARRAY_BUFFER,this.kvBuf);
      gl.bufferSubData(gl.ARRAY_BUFFER,0,this._kvArr);
      drawBatch(this.kvBuf,this.kvN,0.12);
    }
    /* particles additive */
    const parts=[];
    for(const p of this.particles){
      let pos;
      if(p.mtp){const w=(p.s-0.985)*200; pos=add(this.mtpC,[Math.sin(w*3)*8,0,Math.cos(w*3)*6]);}
      else pos=this.pathPoint(Math.min(1,p.s));
      const col=scale(p.col,Math.max(0,p.life));
      this._push(parts,pos,[0.85,0.85],col,4);
    }
    /* (a) per-block forward-pass arrival glow: soft amber sheet over plates
       the token head visited within the last ~0.6 s (additive pass) */
    for(let L=0;L<this.layerAct.length;L++){
      const a=this.layerAct[L];
      if(a>0.03){ const c=plateCenter(L);
        this._push(parts,[c[0],c[2]],[PLATE_W/2,PLATE_H/2],[1.0*a,0.72*a,0.08*a],4); }
    }
    if(parts.length){
      const b=gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER,b);
      gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(parts),gl.DYNAMIC_DRAW);
      gl.depthMask(false); gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA,gl.ONE);
      drawBatch(b,parts.length/8,2.2);
      gl.disable(gl.BLEND); gl.depthMask(true);
      gl.deleteBuffer(b);
    }
  }

  /* bufferSubData only refreshed colour slots; rewrite off/sz/kind once at build */
  _patchDynStatic(){
    if(this._dynStaticFixed)return; this._dynStaticFixed=true;
    /* colours are overwritten in place; offsets/sizes/kinds came from the
       original full upload and remain between sub-updates since we only
       write indices +4..+6 — we must therefore upload stride-8 rows. */
    const dims=this.meta.dims, arr=this._dynArr;
    let k=0;
    for(let L=0;L<dims.layers;L++){
      const c=plateCenter(L);
      const baseX=c[0]-((GRID_C-1)*(CELL+CELL_GAP))/2, baseZ=c[2]+1.9-((GRID_R-1)*(CELL+CELL_GAP))/2;
      for(let r=0;r<GRID_R;r++)for(let cc=0;cc<GRID_C;cc++){
        const o=k*8;
        arr[o]=baseX+cc*(CELL+CELL_GAP); arr[o+1]=baseZ+r*(CELL+CELL_GAP);
        arr[o+2]=CELL/2; arr[o+3]=CELL/2; arr[o+7]=2.0; k++;
      }
    }
  }

  /* ---------------------------- picking/hover ---------------------------- */
  ray(mx,my){
    const cv=this.cv, r=cv.getBoundingClientRect();
    const nx=((mx-r.left)/r.width)*2-1, ny=-(((my-r.top)/r.height)*2-1);
    /* unproject through VP inverse numerically: cast from eye through ndc ray */
    const vp=this.lastVP, eye=this.lastEye;
    /* approximate ray via camera basis (no matrix inverse needed) */
    const z=norm(sub(this.target,eye)), x=[vp[0],vp[4],vp[8]], y=[vp[1],vp[5],vp[9]];
    const dir=norm(add(add(scale(x,nx*Math.tan(0.85/2)*2/ (cv.width/cv.height) ),scale(y,ny*Math.tan(0.85/2)*2)),z));
    return {o:eye,d:dir};
  }
  /* intersect ray with y=0 plane, map to expert cell (exact grid algebra) */
  pickAt(mx,my){
    const {o,d}=this.ray(mx,my);
    if(Math.abs(d[1])<1e-6)return null;
    const t=-o[1]/d[1]; if(t<0)return null;
    const px=o[0]+d[0]*t, pz=o[2]+d[2]*t;
    /* which plate */
    for(const L of this.layers){
      if(Math.abs(px-L.c[0])<=PLATE_W/2 && Math.abs(pz-L.c[2])<=PLATE_H/2){
        const dims=this.meta.dims, c=L.c;
        const bx=c[0]-((GRID_C-1)*(CELL+CELL_GAP))/2, bz=c[2]+1.9-((GRID_R-1)*(CELL+CELL_GAP))/2;
        const cc=Math.round((px-bx)/(CELL+CELL_GAP)), rr=Math.round((pz-bz)/(CELL+CELL_GAP));
        if(cc>=0&&cc<GRID_C&&rr>=0&&rr<GRID_R&&Math.abs(px-(bx+cc*(CELL+CELL_GAP)))<CELL &&
           Math.abs(pz-(bz+rr*(CELL+CELL_GAP)))<CELL){
          const e=rr*GRID_C+cc;
          return {type:"expert",layer:L.i,expert:e};
        }
        return {type:"plate",layer:L.i};
      }
    }
    for(const [c,name] of [[this.embedC,"embedding"],[this.headC,"head"],[this.mtpC,"mtp"]])
      if(Math.abs(px-c[0])<20 && Math.abs(pz-c[2])<14) return {type:name};
    if(Math.abs(px-this.kvC[0])<6 && pz>this.kvC[2]-3 && pz<this.kvC[2]+32*5.3)
      return {type:"kv"};
    return null;
  }
}
