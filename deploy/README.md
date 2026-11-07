# deploy (systemd user unit)

    mkdir -p ~/.config/systemd/user
    cp deploy/strata-atlas.service ~/.config/systemd/user/
    systemctl --user daemon-reload
    systemctl --user enable --now strata-atlas
    # open http://127.0.0.1:8091/

Requirements: engine served on :8080 (`strata serve`), python3.12 (stdlib
only), and tools/build_atlas.py having been run at least once (the
ExecStartPre re-runs it on every start so the atlas tracks the pack).
