# HOIP dataset

Dataset for our BioNLP'24 paper titled ["Mention-Agnostic Information Extraction for Ontological Annotation of Biomedical Articles"](https://aclweb.org/aclwiki/BioNLP_Workshop).

Named entities are typically assumed to appear **explicitly** in text (such textual instances are called *mentions*), and entity features are derived based on the mentions.
Mentions are strong indicators in information extraction tasks, since they directly indicate how entities are described in text.
However, in real-world scenarios, important entities sometimes appear only **implicitly**.

To accelerate the research on **mention-agnostic** information extraction, we introduce **HOIP dataset**, a new biomedical dataset constructed based on [Homeostasis Imbalance Process Ontology (HOIP)](https://github.com/yuki-yamagata/hoip), which focuses on understanding the COVID-19 infectious mechanism (courses).

- HOIP dataset consists of passages (plain text) extracted from PubMed and Wikipedia articles describing biomedical processes in the context of COVID-19 infectious courses. Each passage is a brief portion of an article that describes at least two specific processes. 
- HOIP dataset annotates both entities and relation triples, (head entity, relation, tail entity).
- HOIP dataset requires the capability to infer about entities and relations between them that are not explicitly described, using background knowledge.

The following figure shows an example in the HOIP dataset along with the approach proposed in our paper.
<p align="center>
<img src="https://github.com/norikinishida/hoip-dataset/docs/bionlp2024_figure1.png" width="400">
</p>


For the details of the dataset, please see our paper.

HOIP ontology is also available from the NCBO BioPortal ontology repository site (https://bioportal.bioontology.org/ontologies/HOIP) and GitHub website (https://github.com/yuki-yamagata/hoip).

## Directory structure

<pre>
.
|-- README.md
|-- LICENSE
|-- releases/ # dataset
|   |-- v1/
|       |-- train.json
|       |-- dev.json
|       |-- test.json
|       |--- hoip_ontology.json
|-- construction/ # source codes to generate the dataset
|-- docs/ # our paper and some figures
</pre>

## Citation

If you use the dataset, please cite this paper:

```
@inproceedings{khettari-etal-2024-mention,
    title={Mention-Agnostic Information Extraction for Ontological Annotation of Biomedical Articles},
    author={
        Khettari, Oumaima El and
        Nishida, Noriki and
        Liu, Shanshan and
        Munne, Rumana Ferdous and
        Yamagata, Yuki and
        Quiniou, Solen and
        Chaffron, Samuel and
        Matsumoto, Yuji
    },
    booktitle={The 23rd Workshop on Biomedical Natural Language Processing and BioNLP Shared Tasks},
    month={August},
    year={2024},
    publisher={Association for Computational Linguistics},
    url={},
    doi={}
}
```


