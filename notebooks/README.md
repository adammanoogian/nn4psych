# Notebooks Directory

Jupyter notebooks for exploratory analysis and tutorials.

## Structure

```
notebooks/
├── exploratory/              # Exploratory data analysis
│   ├── plot_pca.ipynb       # PCA visualization
│   ├── plot_pca_contextual.ipynb
│   └── plot_pca_helicopter.ipynb
├── tutorials/                # How-to guides and examples
└── README.md
```

## Guidelines

### Exploratory Notebooks
- Quick analyses and visualizations
- Experimental code that may not be production-ready
- Use for prototyping before moving to scripts/

### Tutorial Notebooks
- Step-by-step guides for users
- Well-documented with markdown explanations
- Should be reproducible with example data

## Best Practices

1. **Clear naming**: Use descriptive filenames (e.g., `analyze_learning_rates.ipynb`)
2. **Top-level summary**: Start with markdown cell explaining notebook purpose
3. **Self-contained**: Include all imports and setup in notebook
4. **Clean outputs**: Clear outputs before committing (optional)
5. **Move to scripts**: Promote stable analysis code to `scripts/analysis/`

## Usage

```bash
# Start Jupyter
jupyter notebook notebooks/

# Or use JupyterLab
jupyter lab notebooks/
```

## Converting to Scripts

When analysis becomes production-ready:
```bash
jupyter nbconvert --to script notebooks/exploratory/my_analysis.ipynb
mv my_analysis.py scripts/analysis/
```
