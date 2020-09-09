# TRAC-LREC2020

## Citation

If you use our work, please cite our short paper [**Offensive Language Detection Explained**](https://hpi.de/fileadmin/user_upload/fachgebiete/naumann/people/risch/risch2020offensive.pdf) as follows:

    @inproceedings{risch2020offensive,
    title = "Offensive Language Detection Explained",
    author = "Risch, Julian and Ruff, Robin and Krestel, Ralf",
    booktitle = "Proceedings of the Second Workshop on Trolling, Aggression and Cyberbullying (TRAC@LREC)",
    year = "2020",
    publisher = "European Language Resources Association (ELRA)",
    pages = "137--143"
    }
    
or our journal article [**Explaining Offensive Language Detection**](https://hpi.de/fileadmin/user_upload/fachgebiete/naumann/people/risch/risch2020explaining.pdf) as follows:

    @article{risch2020explaining,
    author = {Risch, Julian and Ruff, Robin and Krestel, Ralf},
    editor = {Ruppenhofer, Josef and Siegel, Melanie and Struß, Julia Maria},
    journal = {Journal for Language Technology and Computational Linguistics (JLCL)},
    publisher = {German Society for Computational Linguistics and Language Technology (GSCL)}
    volume={34},
    number={1},
    pages={29--47},
    title = {Explaining Offensive Language Detection},
    year = 2020
}

    
## Implementation
The implementation in this repository is part of the Bachelor's thesis by Robin Ruff.

```
.
├── README.md           // This README file
├── classifiers         // Implementations for explainable Text Classifiers
├── datasets.zip            // The modified version of the Toxic Comment dataset, that we used in the thesis (please unzip)
├── environment.yml     // A conda environment file to load all dependencies for this repository
├── jupyter_notebooks   // Includes the jupyter notebooks to train the classifiers and evaluate explanations with EPI and word deletion
├── trained_models.zip      // Includes all trained models as they were used in the thesis evaluations (please unzip)
├── webapp              // Includes a implementation of a simple web app to visualize explanations in the browser 
└── wordvectors.zip         // Includes the GloVe word vectors we used in the thesis (please unzip)
```

### Visualizing Explanations

### 20 Newsgroups

To visualize explanations for 20 newsgroups categorizations in the browser navigate into the `webapp` directory and start the server with `python3 server.py ng`.
Once the server loaded all the models and started open `localhost:8091`.

### Toxic Comments

To visualize explanations for toxic comments categorizations in the browser navigate into the `webapp` directory and start the server with `python3 server.py tc`.
Once the server loaded all the models and started open `localhost:8090`.
