# Updating the original DNN_GSR repository to DeepGS

- Original repository: https://github.com/WenxSong/DNN_GSR
- Current address: https://github.com/WenxSong/DeepGS, default branch `main`

The original repository was renamed first and its files then updated, which preserves the Git history. This document records the steps for repeating such an update.

## 1. Rename the repository

Open the original DNN_GSR repository, go to **Settings → General → Repository name**, enter **DeepGS** and click **Rename**. Links to the old repository redirect to the new address. To keep that redirect working, do not create a new DNN_GSR repository under the same account.

## 2. Update the code

1. Sign in to GitHub Desktop and choose **File → Clone repository → URL**. Enter https://github.com/WenxSong/DeepGS and clone the `main` branch into a new empty folder, for example `E:\DeepGS_publish`.
2. In the clone, remove the old `data/`, `step1_auto_diff.py`, `step1_finite_diff.py` and `step2_sparse_regression.py`. The old version stays in the Git history. Keep the hidden `.git/`.
3. Open `intermediate\github_upload_files` in this working folder and copy all of its contents into the clone. This is the publication copy with local materials excluded; it includes `.github/` and `.gitignore`. Keep the clone's own `.git/`.
4. Return to GitHub Desktop and review **Changes**. Enter the commit message `Update DNN-GSR to DeepGS Toolkit`, click **Commit to main**, then **Push origin**.

If you already have a clone of the original repository, run this once in that clone, then follow the update and commit steps above:

```bash
git remote set-url origin https://github.com/WenxSong/DeepGS.git
```

This command cannot run directly in a source working folder without `.git`.

Copying over files of the same name does not remove old scripts, and `.gitignore` does not stop tracking old files, so check that the old files are deleted. A normal commit preserves the history; no force-push or repository rebuild is needed.

## 3. Check and release

Check the GitHub home page and Actions. Once CI has finished, you can create a v0.1.0 Release and attach the wheel and source package from `intermediate/releases/current/`.

The software's display name stays **DeepGS Toolkit**, and the Python package and command are both **deepgs**. The DNN-GSR name and paper citations are retained. The [original code before the update](https://github.com/WenxSong/DeepGS/tree/32058cc16ae3493598d7a775f75d8b390eda4c01) remains reachable by the same commit hash after the rename.

Suggested About description:

> DeepGS Toolkit: sparse structure selection and physics-informed coefficient inversion for soil water flow equations.

The README gives the quick-start commands, and the user guide has the detailed options.

Official documentation: [renaming a repository](https://docs.github.com/en/repositories/creating-and-managing-repositories/renaming-a-repository), [cloning with GitHub Desktop](https://docs.github.com/en/desktop/adding-and-cloning-repositories/cloning-and-forking-repositories-from-github-desktop), [pushing commits](https://docs.github.com/en/desktop/making-changes-in-a-branch/pushing-changes-to-github-from-github-desktop).
