# mlpack

- **Courses.** Clone the repo or download the ZIP, then open [courses/index.html](courses/index.html) in a browser. Start with the topic you want to review and follow its chapter links.
- **Practice.** Try the [TorchCode templates](practice/TorchCode/templates) or pick a problem from the [TorchLeet problem list](practice/TorchLeet/PROBLEMS.md). The [comparison notebook](practice/TorchCode/templates/00_torchcode_vs_torchleet.ipynb) shows where the two sets overlap. Both projects include their original reference solutions.
- **PyTorch cheat sheet.** [Read the Markdown](reference/pytorch-cheat-sheet.md) or [download the three-page PDF](reference/pytorch-cheat-sheet.pdf).

## Run the notebooks

Install JupyterLab and the bundled TorchCode judge from the repo root. Open an exercise in `practice/TorchCode/templates` and run its check cell after writing your implementation.

```sh
python -m pip install jupyterlab -e ./practice/TorchCode
jupyter lab
```

For TorchLeet, see its [quick start](practice/TorchLeet/README.md#quick-start) and the setup notes in each exercise.

## Sources

The HTML courses and PyTorch cheat sheet come from Learn ML. The practice projects use the following upstream revisions.

- [TorchCode by duoan](https://github.com/duoan/TorchCode/tree/e3c8782a4be4f7cdb4de871ec739f66370968a77). The upstream [package metadata](practice/TorchCode/pyproject.toml) declares MIT. The revision has no separate license file.
- [TorchLeet by Chandrahas Aroori and contributors](https://github.com/Exorust/TorchLeet/tree/dac1e913d5291f4c21c636af80092a3c38b472ba). See the bundled [MIT license](practice/TorchLeet/LICENSE).
