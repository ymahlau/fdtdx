# Download the repository as a zip file
curl -L https://github.com/ymahlau/fdtdx-notebooks/archive/refs/heads/main.zip -o repo.zip

# Extract to notebooks folder
unzip repo.zip
mkdir -p docs/source/notebooks/
mv fdtdx-notebooks-main/*.ipynb docs/source/notebooks/
mv fdtdx-notebooks-main/quickstart/*.ipynb docs/source/notebooks/
mv fdtdx-notebooks-main/advanced/*.ipynb docs/source/notebooks/

# Clean up
rm repo.zip
rm -rf fdtdx-notebooks-main
