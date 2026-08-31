# TO-DO list and package build instructions

## To-Do list (To be removed for release)

- Routine gateway backend and operations-related routines (e.g., data inventory and
  scheduling) are NOT included in this package.
- Update outdated or missing docstrings
- Remove/rectify FIXME and TODO items
- Refactor for clarity
- Update README to be more useful
- Update routines to use the new file naming conventions that include the orbit number
  when that comes to pass (needs to be coordinated with Brent for gateway, on hold).
- Add option to define option values with *implicit* evars (set in environment) rather
  than *explicit* evars (passed in with the option flag, e.g., -d0 $START_DATE)
- Add a multiprocessing pool option to speed up plot generation?

## Build/install instructions (Temporary notes to self, also to be removed for release)

- Configure testpypi credentials

```bash
[distutils]
index-servers =
    testpypi

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-AgENdGVzdC5weXBpLm9yZwIkZWQ4M2I5OWUtYWRjMC00NzU3LThjZjItODJiYjA1NTBhNDA1AAIqWzMsIjJhNDY0YTYwLWNiNDMtNGE0NS1hOTRhLWYxZDRmOGJiNDhlNCJdAAAGIOlJ-CNtUdgwz2mIYAI0bFbCpby9eW1gek6rTVspdAqC
```

- In ezieview directory:

```bash
uv venv -p 3.12
uv pip install --upgrade build
uv pip install --upgrade twine
. .venv/bin/activate
UV_CACHE_DIR=/project/ezie/.cache/uv PIP_INDEX_URL=https://pypi.org/simple python3 -m build
```

or

```bash
uv build --index https://pypi.org/simple
```

## Upload to testpypi with twine or uv

Publish/upload the package using `uv publish`:

```bash
uv publish dist/ezieview-0.1.2-py3-none-any.whl \
    --system-certs \
    --verbose \
    --publish-url https://test.pypi.org/legacy \
    -u __token__ \
    -p pypi-AgENdGVzdC5weXBpLm9yZwIkZWQ4M2I5OWUtYWRjMC00NzU3LThjZjItODJiYjA1NTBhNDA1AAIqWzMsIjJhNDY0YTYwLWNiNDMtNGE0NS1hOTRhLWYxZDRmOGJiNDhlNCJdAAAGIOlJ-CNtUdgwz2mIYAI0bFbCpby9eW1gek6rTVspdAqC
```

or alternatively using `twine upload`:

```bash
twine upload dist/ezieview-0.1.2-py3-none-any.whl \
    --verbose \
    --repository-url https://test.pypi.org/legacy \
    -u __token__ \
    -p pypi-AgENdGVzdC5weXBpLm9yZwIkZWQ4M2I5OWUtYWRjMC00NzU3LThjZjItODJiYjA1NTBhNDA1AAIqWzMsIjJhNDY0YTYwLWNiNDMtNGE0NS1hOTRhLWYxZDRmOGJiNDhlNCJdAAAGIOlJ-CNtUdgwz2mIYAI0bFbCpby9eW1gek6rTVspdAqC 
    ```

## Test for successful upload

```bash
mkdir ezvtest
cd ezvtest
uv venv -p 3.12
. .venv/bin/activate
uv pip install \
    --system-certs \
    --default-index "https://test.pypi.org/simple" \
    --index "https://pypi.org/simple" \
    ezieview==0.1.2
view_coverage --help
view_orbits --help
```

- Note that specifying a non-standard location for UV_CACHE_DIR is only useful on DMZ
  systems where user HOME directory space is limited and the uv cache directory may also
  be mounted on a different disk volume than the one where the code and venv are
  installed.

- The `--system-certs` flag may be helpful when working on APLNIS and accessing the
  outside world through the APL proxy.

- Note also that the use of setup.py/setuptools is now deprecated.

### Install ezieView

- Create the directory where you'd like to install the package and initialize your venv:

```bash
mkdir ezvtest
cd ezvtest
uv venv -p 3.12
```

- Option A, for developers: Install in "editable" mode, allowing you to modify the
  package code and see the changes immediately. First `git clone` the repo from
  `aplkaiju`, then `cd` nto the repo directory and run the following command:

```bash
[UV_CACHE_DIR=/project/ezie/.cache/uv] uv pip install \
    [--system-certs]  \
    --index "https://pypi.org/simple" -e  .
```


- Option B, for regular users: Install in the normal fashion. Eventually the package
  will be available from regular PyPi and will be
  installed by name rather than a local file path/wheel.

```bash
UV_CACHE_DIR=/project/ezie/.cache/uv uv pip install ../ezieView/dist/ezieview-0.0.2-py3-none-any.whl
```
