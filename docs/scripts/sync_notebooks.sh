# Download the repository as a zip file
curl -L https://github.com/ymahlau/fdtdx-notebooks/archive/refs/heads/main.zip -o repo.zip

# Extract to notebooks folder
unzip repo.zip
mkdir -p docs/source/notebooks/
mkdir -p docs/source/notebooks/quickstart
mkdir -p docs/source/notebooks/advanced
mkdir -p docs/source/notebooks/components

mv fdtdx-notebooks-main/quickstart/*.ipynb docs/source/notebooks/quickstart
mv fdtdx-notebooks-main/advanced/*.ipynb docs/source/notebooks/advanced
mv fdtdx-notebooks-main/components/*.ipynb docs/source/notebooks/components

# Clean up
rm repo.zip
rm -rf fdtdx-notebooks-main
