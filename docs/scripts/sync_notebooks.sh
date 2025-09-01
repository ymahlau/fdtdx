# Download the repository as a zip file
curl -L https://github.com/ymahlau/fdtdx-notebooks/archive/refs/heads/main.zip -o repo.zip

# Extract to notebooks folder
unzip repo.zip
mkdir -p docs/source/_static/notebooks/
mv fdtdx-notebooks-main/*.ipynb docs/source/_static/notebooks/

# Clean up
rm repo.zip
rm -rf fdtdx-notebooks-main
