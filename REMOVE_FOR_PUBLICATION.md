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

- Configure artifactory/pypi credentials
```o[distutils]
index-servers = python-remote

[python-remote]
repository: https://artifactory.jhuapl.edu/artifactory/api/pypi/python-remote
username: YOUR_USERNAME
password: YOUR_API_KEY_OR_PASSWORD
```

- In ezieview directory:

```bash
uv venv -p 3.12
uv pip install --upgrade build
uv pip install --upgrade twine
. .venv/bin/activate
UV_CACHE_DIR=/project/ezie/.cache/uv PIP_INDEX_URL=https://pypi.org/simple python3 -m build

# Upload to artifactory with twine
twine upload --repository apl ezieview/dist/*

# Test for successful upload
mkdir ezvtest
cd ezvtest
# python -m venv /tmp/test_venv
# source /tmp/test_venv/bin/activate
uv venv -p 3.12
. .venv/bin/activate

python -m pip install --index-url \
    "https://artifactory.jhuapl.edu/artifactory/api/pypi/pypi-apl-virtual/simple" \
    --no-deps ezieview==0.1.0

```

- Note that specifying a non-standard location for UV_CACHE_DIR is only useful on DMZ
  systems where user HOME directory space is limited and the uv cache directory may also
  be mounted on a different disk volume than the one where the code and venv are
  installed.

- Note also that the use of setup.py/setuptools is now deprecated.

### Install ezieView

- Create the directory where you'd like to install the package and initialize your venv:

```bash
mkdir ezvtest
cd ezvtest
uv venv -p 3.12
```

- Option A, for developers: Install in "editable" mode, allowing you to modify the
  package code and see the changes immediately:

```bash
UV_CACHE_DIR=/project/ezie/.cache/uv uv pip install -e ../ezieView
```

- Option B, for regular users: Install in the normal fashion. Eventually the package
  will be available from the Artifactory (JHUAPL) or PyPi (rest of world) and will be
  installed by name rather than a local file path/wheel.

```bash
UV_CACHE_DIR=/project/ezie/.cache/uv uv pip install ../ezieView/dist/ezieview-0.0.1-py3-none-any.whl
```