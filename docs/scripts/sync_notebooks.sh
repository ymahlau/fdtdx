# Download the repository as a zip file
curl -L https://github.com/ymahlau/fdtdx-notebooks/archive/refs/heads/main.zip -o repo.zip

# Extract to notebooks folder
unzip repo.zip
mkdir -p docs/source/notebooks/
mkdir -p docs/source/notebooks/quickstart
mkdir -p docs/source/notebooks/physics
mkdir -p docs/source/notebooks/components
mkdir -p docs/source/notebooks/advanced

# Remove notebooks that no longer exist upstream, so deleted/renamed
# notebooks don't leave stale pages or stale cards behind
rm -f docs/source/notebooks/quickstart/*.ipynb
rm -f docs/source/notebooks/physics/*.ipynb
rm -f docs/source/notebooks/components/*.ipynb
rm -f docs/source/notebooks/advanced/*.ipynb

mv fdtdx-notebooks-main/quickstart/*.ipynb docs/source/notebooks/quickstart
mv fdtdx-notebooks-main/physics/*.ipynb docs/source/notebooks/physics
mv fdtdx-notebooks-main/components/*.ipynb docs/source/notebooks/components
mv fdtdx-notebooks-main/advanced/*.ipynb docs/source/notebooks/advanced

# Clean up
rm repo.zip
rm -rf fdtdx-notebooks-main

# Regenerate the notebook gallery cards on each landing page
python3 docs/source/generate_cards.py
