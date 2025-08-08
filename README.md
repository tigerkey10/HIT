# HIT (Hypergraph Interaction Transformer)
> ### AI-driven therapeutic gene target prediction
> An Explainable AI model for fast and precise identification of therapeutic gene candidates
> by integrating complex disease-gene relationships and ontology information.

<img width="1438" height="807" alt="hit_figures" src="https://github.com/user-attachments/assets/77f3116c-87ab-4b54-9f66-59ed4c05e8eb" />



## 🚀 Key Features
✅ Hypergraph-based modeling: Captures many-to-many relationships between diseases and genes.

✅ Ontology integration: Utilizes disease and gene ontology information for enhanced representation.

✅ Explainable AI: Provides interpretable insights into model decision-making.

✅ Scalable implementation: Built on PyTorch, designed for large-scale biomedical datasets.

## 📂 Project Structure
```bash
HIT/
├── datasets/          # Original datasets
├── models/            # Model implementation
├── exp.py             # Main execution script
├── dataset.py         # Dataset construction script
├── trainer.py         # Model trainer
├── utils.py           # utils
├── requirements.txt   # Python dependencies
└── README.md
```

## ⚙️ Installation
1. Clone this repository:
```bash
$ git clone https://github.com/tigerkey10/HIT.git
$ cd HIT
```
2. Install required dependencies:
```bash
$ pip install -r requirements.txt
```

## ▶️ Usage
Run the model:
```bash
$ python exp.py 
```
Run with custom arguments:
```bash
# Example
$ python exp.py --epochs 50 --lr 1e-3
```

## 🌐 Webserver Access
You can access the deployed HIT webserver interface via the link below:

🔗 [HIT Webserver](http://mlblabhit.org/)


### Webserver interface main
<img width="1851" height="1125" alt="image" src="https://github.com/user-attachments/assets/2b97d217-068b-4d78-a6ca-ef459393dcd4" />


### Results example
<img width="1286" height="1853" alt="image" src="https://github.com/user-attachments/assets/d92ad941-c7ea-4d12-982d-429d6148e4ad" />

### Feedback
<img width="1888" height="1155" alt="image" src="https://github.com/user-attachments/assets/8961cb81-b65a-44c7-bd2c-018613028cba" />






## 📖 Reference
Kim, Kibeom, et al. "Therapeutic gene target prediction using novel deep hypergraph representation learning." Briefings in Bioinformatics 26.1 (Jan 2025).
🔗 [Paper](https://doi.org/10.1093/bib/bbaf019)

💡 If you use this code for research, please cite the above paper.

## 📜 License
This project is licensed under the MIT License.
































