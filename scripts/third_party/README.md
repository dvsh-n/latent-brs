# Local Third-Party Changes

The repositories under `third_party/` are upstream projects managed as Git
submodules. The parent repository can only record a submodule commit SHA; it
cannot directly record dirty files inside a submodule.

This directory preserves local code and configuration changes as patch files in
the `latent-brs` repository. Generated outputs, datasets, checkpoints, logs, and
caches are intentionally excluded.

## Protect Upstream Repositories

Run this once after cloning or initializing submodules:

```bash
scripts/third_party/protect_upstream_pushes.sh
```

This keeps each submodule's fetch URL unchanged while setting its push URL to an
invalid `disabled://push-protected` URL. Fetching upstream still works, but an
accidental `git push origin ...` from a submodule fails.

The protection is stored in local Git configuration and must be rerun on each
new clone.

## Save Local Changes

Refresh the tracked patch snapshots:

```bash
scripts/third_party/save_local_patches.sh
git add third_party_patches/
```

Review the patch diff before committing it. The save script includes tracked
changes plus a small reviewed list of untracked source/config files.

## Apply Local Changes

After initializing clean submodules, apply the saved patches:

```bash
scripts/third_party/apply_local_patches.sh
```

The apply script does not reset or discard existing submodule changes. Apply
patches to clean submodule checkouts whenever possible.
