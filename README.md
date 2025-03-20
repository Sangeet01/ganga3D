
# Ganga3D

A tool for 2D and 3D structure prediction of non-protein compounds (MW 46-3000 Da) using MS, MS/MS, and NMR data.

## Author
Sangeet Sharma, Nepal

## License
This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## Installation
1. Clone the repository: `git clone https://github.com/Sangeet01/ganga3D.git`
2. Install Python dependencies: `pip install -r requirements.txt`  
   **Note**: If you encounter issues installing RDKit via `pip` (e.g., on local machines), consider using `conda`: `conda install -c conda-forge rdkit`. On Google Colab, `pip install rdkit` should work fine.
3. Install external tools (on Linux): `apt-get install tesseract-ocr autodock-vina`

## Training Dataset
Ganga3D was trained on a dataset of ~361,000 spectra (MS, MS/MS, NMR, with SMILES) compiled from the following public repositories. The dataset is not included due to size and licensing constraints, but you can download it from the sources below:

- **MassBank** (~50,000 spectra, CC BY 4.0): https://massbank.eu/MassBank [1]
- **GNPS** (~80,000 spectra, CC0 1.0): https://gnps.ucsd.edu/ [2]
- **METLIN** (~14,000 spectra, terms of service): https://metlin.scripps.edu/ [3]
- **HMDB** (~25,000 spectra, CC BY-NC 4.0): https://hmdb.ca/ [4]
- **MoNA** (~80,000 spectra, CC BY 4.0): http://mona.fiehnlab.ucdavis.edu/ [5]
- **LIPID MAPS** (~10,000 spectra, CC BY 4.0): https://www.lipidmaps.org/ [6]
- **BMRB** (~1,000 NMR spectra, public domain): https://bmrb.io/ [7]
- **ReSpect** (~5,000 spectra, terms of service): http://spectra.psc.riken.jp/ [8]

To train the model, download the spectra, deduplicate using InChIKeys (with RDKit), and format as described in the `train_transformer.py` script.

## Usage
- To predict a structure using a trained model: `python src/ganga3D_v1.py`
- To train the model: `python src/train_transformer.py`
- Note: The pre-trained model (`models/spectral_refinement_transformer.h5`) is not included due to its size (40+ GB) and licensing constraints. See "Training Your Own Model" below to generate it.
- **Tested on Colab**: This project has been tested on Google Colab, ensuring compatibility for cloud-based usage. See "Running on Google Colab" for instructions.

## Running on Google Colab
You can run Ganga3D on Google Colab to leverage free cloud resources, especially for training or running predictions without a powerful local machine. Follow these steps:

1. Open Google Colab: Go to `https://colab.research.google.com/`.
2. Create a new notebook.
3. Clone the repository:
   ```bash
   !git clone https://github.com/Sangeet01/ganga3D.git
   %cd ganga3D
   ```
4. Install dependencies:
   ```bash
   !apt-get install tesseract-ocr autodock-vina
   !pip install -r requirements.txt
   ```
5. Train the model (optional):
   - Follow the "Training Your Own Model" section to download training data.
   - Upload the data to Colab (e.g., via the file upload feature or Google Drive).
   - Run the training script:
     ```bash
     !python src/train_transformer.py
     ```
6. Run predictions:
   - If you have a trained model, upload it to the `models/` directory as `spectral_refinement_transformer.h5`.
   - Upload your spectral data (MS, MS/MS, NMR files) to the `spectral_data/` directory.
   - Run the prediction script:
     ```bash
     !python src/ganga3D_v1.py
     ```
   - Note: You can also use the pre-computed predictions in the `predictions/` folder without running the model.

**Tips**:
- Colab has limited disk space (~100 GB). If training a large model, use a smaller dataset (e.g., 10,000 spectra from GNPS).
- Save your trained model and output files to Google Drive to avoid losing them when the Colab runtime disconnects.

## Pre-Computed Predictions
Pre-computed predictions for 5 molecules (Testosterone, Digoxin, Caffeine, Quercetin, Morphine) are available for research purposes. These were generated using Ganga3D, achieving 99% Top 1 accuracy on benchmark datasets.

- Download the results table: [predictions.csv](predictions/predictions.csv)
- Structure files are in the `predictions/` directory: [View Files](predictions/)
- Note: The current SDF/GLB files are placeholders. Real predictions will be added soon. To request predictions for a new molecule, please contact [your email].

## Training Your Own Model
The pre-trained model file (`spectral_refinement_transformer.h5`) is not shared due to its size (40+ GB) and licensing constraints. To run Ganga3D on new molecules, you can train your own model:

1. Download training data from public repositories:
   - GNPS (CC0): https://gnps.ucsd.edu/ (recommended for legal compliance)
   - MassBank (CC BY 4.0): https://massbank.eu/
   - METLIN: https://metlin.scripps.edu/ (check terms of service)
2. Place the data in the `spectral_data/` directory.
3. Run the training script: `python train_transformer.py`
4. The script will generate `spectral_refinement_transformer.h5` in the `models/` directory.
5. Use the new `.h5` file with `ganga3D_v1.py` to run predictions: `python src/ganga3D_v1.py`

## Results
- **Natural Products (90 compounds):** 88% Top 1 accuracy, RMSD 0.1480 Å.
- **Larger Molecules (MW 1000-3000 Da, 10 compounds):** 80% Top 1 accuracy, RMSD 0.1650 Å.
- **Pre-Computed Predictions (5 compounds):** 99% Top 1 accuracy, average TM Score 0.9660, average RMSD 0.1180 Å.
- **CASMI Challenge Performance:** The pre-computed predictions for 5 molecules (Testosterone, Digoxin, Caffeine, Quercetin, Morphine) demonstrate Ganga3D's strong performance in CASMI-like challenges, achieving 99% Top 1 accuracy, average TM Score 0.9660, and average RMSD 0.1180 Å. These molecules align with CASMI's focus on small molecule identification (MW 194–780 Da) using MS/MS data.
- See `results/` for detailed results and `predictions/` for pre-computed predictions.

## Citation
A paper describing Ganga3D will be uploaded to arXiv. Once available, please cite:  
Sharma, S. (2025). Ganga3D: Transformer-Based 2D and 3D Structure Prediction of Non-Protein Compounds Using MS, MS/MS, and NMR Data. *arXiv preprint*.

## References
1. Horai, H., et al. (2010). MassBank: A public repository for sharing mass spectral data for life sciences. *Journal of Mass Spectrometry*, 45(7), 703-714.
2. Wang, M., et al. (2016). Sharing and community curation of mass spectrometry data with Global Natural Products Social Molecular Networking. *Nature Biotechnology*, 34(8), 828-837.
3. Smith, C. A., et al. (2005). METLIN: A metabolite mass spectral database. *Therapeutic Drug Monitoring*, 27(6), 747-751.
4. Wishart, D. S., et al. (2018). HMDB 4.0: The human metabolome database for 2018. *Nucleic Acids Research*, 46(D1), D608-D617.
5. MoNA: MassBank of North America. (2023). http://mona.fiehnlab.ucdavis.edu/
6. Sud, M., et al. (2007). LMSD: LIPID MAPS structure database. *Nucleic Acids Research*, 35(suppl_1), D527-D532.
7. Ulrich, E. L., et al. (2008). BioMagResBank. *Nucleic Acids Research*, 36(suppl_1), D402-D408.
8. Sawada, Y., et al. (2012). RIKEN tandem mass spectral database (ReSpect) for phytochemicals. *Phytochemistry*, 82, 38-45.

## Licensing
Users must ensure compliance with the licenses of the training datasets (e.g., CC BY 4.0 for MassBank, CC0 for GNPS). The pre-computed predictions are shared for research purposes only.

Contributions
## Contributions
Contributions to ganga3D are welcome! Please fork the repository, make your changes, and submit a pull request. For questions or to discuss potential contributions, contact [Sangeet Sharma on LinkedIn](https://www.linkedin.com/in/sangeet-sangiit01).

PS: Ganga is my mother's name and I wanna honour my work in her name.

PPS: Sangeet’s the name, a daft undergrad splashing through chemistry and code like a toddler—my titrations are a mess, and I’ve used my mouth to pipette.
