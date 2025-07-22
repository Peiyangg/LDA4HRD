import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import umap
    import hdbscan
    import pandas 
    import numpy as np
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""## HDBSCAN clustering enhanced by UMAP""")
    return


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(r"""## Plot metrics""")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
