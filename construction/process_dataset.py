import argparse
from collections import defaultdict
import copy
import io
import json
import os

import numpy as np
import pandas as pd
from tqdm import tqdm


COLUMNS = [
    "course",
    "course_label",
    "lv2", # Subject entity ID
    "lv2_label", # Subject entity name
    "relation", # Relation ID
    "relation_type", # Relation name
    "lv2_target", # Object entity ID
    "lv2_target_label", # Object entity name
    "pubmedID",
    "description"
]

FINE_GRAINED_COLUMNS = [
    "process", # Subject entity ID
    "process_label", # Subject entity name
    "target", # Object entity ID
    "target_label", # Object entity name
    "context_target", # Context ID
    "context_target_label", # Context name
    "participant_type" # Object entity type
]


def main(args):
    # Process the arguments
    path_input_file = args.input_file
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    path_output_dir = args.output_dir

    # Get triples from the original CSV file
    print(f"Building triples from {path_input_file} ...")
    triples = get_triples(path_input_file=path_input_file) # list[Triple]

    # Aggregate triples into a document level based on the PubMed ID and the text
    print("Aggregating triples into documents ...")
    documents = aggregate_triples(triples=triples) # list[Doc]

    # Merge annotations
    print("Merging triples ...")
    documents = merge_triples(documents=documents)

    # print("Assigning hypernym-triple marks ...")
    # entity_dict = read_json(path=args.ontology)
    # documents = assign_hypernym_marks(
    #     documents=documents,
    #     entity_dict=entity_dict
    # )

    # print("Transforming to KAPipe format ...")
    # documents = transform_to_kapipe_format(documents)

    print("Splitting and saving the documents ...")
    train_documents, dev_documents, test_documents = split_documents(
        documents=documents,
        n_dev=30,
        n_test=30
    )

    write_json(path_output_dir + "/train.json", train_documents)
    write_json(path_output_dir + "/dev.json", dev_documents)
    write_json(path_output_dir + "/test.json", test_documents)

    print("Done.")


# ---------------------------


def get_triples(path_input_file):
    """Get triples (dictionary) from the original CSV file

    Parameters
    ----------
    path_input_file: str

    Returns
    -------
    Dict[str, Triple]
    """

    df = read_csv(path_input_file, with_head=True, with_id=False, delimiter=",")

    # Drop records (i.e., triples) with NaN values for target attributes
    df = df.dropna(subset=COLUMNS)

    # Extract columns of interest
    df = df.loc[:, COLUMNS + FINE_GRAINED_COLUMNS]

    # Transform DataFrame to dictionary
    dct = df.to_dict()
    key = list(dct.keys())[0]
    triple_keys = list(dct[key].keys())

    # Create a list of triples: [Triple]
    triples = []
    for triple_key in triple_keys:
        # Key
        pubmed_id = dct["pubmedID"][triple_key]
        # Text
        text = dct["description"][triple_key]
        # Head entity
        head_entity_id = dct["lv2"][triple_key]
        head_entity_name = dct["lv2_label"][triple_key]
        head_entity_type = "process" # TODO: Correct?
        fg_head_entity_id = dct["process"][triple_key]
        fg_head_entity_name = dct["process_label"][triple_key]
        fg_head_entity_type = "process"
        # Tail entity
        tail_entity_id = dct["lv2_target"][triple_key]
        tail_entity_name = dct["lv2_target_label"][triple_key]
        tail_entity_type = dct["participant_type"][triple_key]
        fg_tail_entity_id = dct["target"][triple_key]
        fg_tail_entity_name = dct["target_label"][triple_key]
        fg_tail_entity_type = dct["participant_type"][triple_key]
        # Fix the naming issue:
        #   Entity definition is wrongly annotated as the canonical name.
        # head_entity_name = fix_name(name=head_entity_name)
        # fg_head_entity_name = fix_name(name=fg_head_entity_name)
        tail_entity_name = fix_name(name=tail_entity_name)
        fg_tail_entity_name = fix_name(name=fg_tail_entity_name)
        # Relation
        relation_id = dct["relation"][triple_key]
        relation_name = dct["relation_type"][triple_key]
        # Context
        context_id = dct["context_target"][triple_key]
        context_name = dct["context_target_label"][triple_key]
        # Cource
        course_id = dct["course"][triple_key]
        course_name = dct["course_label"][triple_key]

        triple = {
            "triple_key": triple_key,
            "pubmed_id": pubmed_id,
            "text": text.replace("\\n", " ").rstrip(),
            "sentences": text.rstrip().split("\\n"),
            #
            "head_entity": {"id": head_entity_id,
                            "name": head_entity_name,
                            "type": head_entity_type},
            "fg_head_entity": {"id": fg_head_entity_id,
                               "name": fg_head_entity_name,
                               "type": fg_head_entity_type},
            "tail_entity": {"id": tail_entity_id,
                            "name": tail_entity_name,
                            "type": tail_entity_type},
            "fg_tail_entity": {"id": fg_tail_entity_id,
                               "name": fg_tail_entity_name,
                               "type": fg_tail_entity_type},
            "relation": {"id": relation_id,
                         "name": relation_name},
            "context": {"id": context_id,
                        "name": context_name},
            "course": {"id": course_id,
                       "name": course_name},
        }
        triples.append(triple)

    return triples


def fix_name(name):
    if name.startswith(
        "The movement of a macrophage in response to an external stimulus."
    ):
        new_name = "macrophage chemotaxis"
        print(f"Fixed name from '{name}' to '{new_name}'")
        return new_name
    else:
        return name


# ---------------------------


def aggregate_triples(triples):
    """Aggregate triples based on the PubMed ID and the text. Triples with the same PubMed ID and the text are treated as a single document instance.

    Parameters
    ----------
    triples : List[Triple]

    Returns
    -------
    list[Doc]
    """
    # Aggregate triples based on the Pubmed ID and the text
    key_to_triples = {}
    for triple in triples:
        pubmed_id = triple["pubmed_id"]
        text = triple["text"]
        if not (pubmed_id, text) in key_to_triples:
            key_to_triples[(pubmed_id, text)] = [triple]
        else:
            key_to_triples[(pubmed_id, text)].append(triple)

    # We treat triples of the same Pubmed ID and the text as a single data
    documents = []
    for data_i, (pubmed_id, text) in enumerate(key_to_triples.keys()):
        doc_key = f"ID{data_i}"

        triples = key_to_triples[(pubmed_id, text)]
        sentences = triples[0]["sentences"]

        # pubmed_id = triples[0]["pubmed_id"]
        # for i in range(len(triple_list)):
        #     assert triples[i]["pubmed_id"] == pubmed_id

        xs = []
        for triple in triples:
            x = {"head_entity": triple["head_entity"],
                 "fg_head_entity": triple["fg_head_entity"],
                 "tail_entity": triple["tail_entity"],
                 "fg_tail_entity": triple["fg_tail_entity"],
                 "relation": triple["relation"],
                 "context": triple["context"],
                 "course": triple["course"],
                 "merged": False}
            xs.append(x)
        triples = xs

        data = {
            "doc_key": doc_key,
            "pubmed_id": pubmed_id,
            "sentences": sentences,
            "triples": triples,
        }
        documents.append(data)

    return documents


# ---------------------------


def merge_triples(documents):
    """If one passage is completely contained within another, the triples associated with the shorter passage are merged with the triples associated with the longer passage.

    Parameters
    ----------
    documents : list[Doc]

    Returns
    -------
    list[Doc]
    """
    # Step 1. Get contained->containers edges
    edges = get_edges(documents=documents)

    # Step 2. Collect terminal nodes
    terminals, nonterminals = get_terminal_nodes(edges=edges)
    # print(terminals)
    while len(terminals) != 0:
        # Step 3. For each terminal node, merge the triples of it to the longer documents that completely contain it
        for src_i in terminals:
            for dst_i in edges[src_i]:
                documents[dst_i] = merge_triples_for_two_documents(
                    src_document=documents[src_i],
                    dst_document=documents[dst_i]
                )
                # print(f"Merged from {src_key} to {dst_key}")
        # Step 4. Remove the terminal nodes in `edges`
        for src_i in terminals:
            edges.pop(src_i)
        # Step 2 (re). Re-collect terminal nodes
        terminals, nonterminals = get_terminal_nodes(edges=edges)
        # print(terminals)

    return documents


def get_edges(documents):
    """Get a dictionary of type Dict[int, List[int]], where the key corresponds to a document index and the value is the list of document indices whose text contains the source document text completely.

    Parameters
    ----------
    documents : list[Doc]

    Returns
    -------
    Dict[int, List[int]]
    """
    src2dsts = {}
    for doc1_i in range(len(documents)):
        document1 = documents[doc1_i]
        text1 = " ".join(document1["sentences"])
        for doc2_i in range(len(documents)):
            if doc1_i == doc2_i:
                continue
            document2 = documents[doc2_i]
            text2 = " ".join(document2["sentences"])
            if text1 in text2:
                if not doc1_i in src2dsts:
                    src2dsts[doc1_i] = [doc2_i]
                else:
                    src2dsts[doc1_i].append(doc2_i)
    # dst2srcs = {}
    # for src_key in src2dsts.keys():
    #     dst2srcs[src_key] = []
    #     for dst_key in src2dsts[src_key]:
    #         dst2srcs[dst_key] = []
    # for src_key in src2dsts.keys():
    #     for dst_key in src2dsts[src_key]:
    #         dst2srcs[dst_key].append(src_key)
    # return src2dsts, dst2srcs
    return src2dsts


def get_terminal_nodes(edges):
    """Find terminal nodes (i.e., nodes without any in-coming edges) and nonterminal nodes (with in-coming edges)

    Parameters
    ----------
    edges : Dict[int, List[int]]

    Returns
    -------
    List[int], List[int]
    """
    terminal_nodes = []
    nonterminal_nodes = []
    for key in edges.keys():
        pointed_by_others = False
        # Check whether `key` is pointed by any other node.
        # If `key` is NOT a dst key of any other node,
        # we treat the `key` as the terminal node
        for src_key in edges.keys():
            if key == src_key:
                continue
            dst_keys = edges[src_key]
            if key in dst_keys:
                pointed_by_others = True
                break
        if not pointed_by_others:
            terminal_nodes.append(key)
        else:
            nonterminal_nodes.append(key)
    return terminal_nodes, nonterminal_nodes


def merge_triples_for_two_documents(src_document, dst_document):
    """Merge annotations (triples) of two document instances

    Parameters
    ----------
    src_document : Doc (i.e., Dict[str, Any])
    dst_document : Doc

    Returns
    -------
    Doc
    """
    # dst_document = copy.deepcopy(dst_document)
    for triple in src_document["triples"]:
        if not is_contained(triple=triple, triples=dst_document["triples"]):
            triple = copy.deepcopy(triple)
            triple["merged"] = True
            dst_document["triples"].append(triple)
    return dst_document


def is_contained(triple, triples):
    """Returns True if the arg1 `triple` is contained in the arg2 `triples`, otherwise False

    Parameters
    ----------
    triple : Triple (i.e., Dict[str, Any])
    triples : List[Triple]

    Returns
    -------
    bool
    """
    for i in range(len(triples)):
        if (
            triple["head_entity"]["name"] == triples[i]["head_entity"]["name"]
            and
            triple["tail_entity"]["name"] == triples[i]["tail_entity"]["name"]
            and
            triple["relation"]["name"] == triples[i]["relation"]["name"]
        ):
            return True
    return False


# ---------------------------


def assign_hypernym_marks(documents, entity_dict):
    # Build a map from entity ID to a list of tree numbers
    entity_id_to_tree_numbers = {
        epage["Class ID"]: epage["TreeNumbers"] for epage in entity_dict
    }

    # For each triple, we check whether the hypernym version is involved in the annotation.
    # Hypernym triples are filtered out as the annotation.
    # Hypernym ENTITIES are kept.
    # Examples:
    #   - Original triples: (A, has_result, B), (A, has_result, b), (a, has_result, B), (a, has_result, b), ...
    #   - Original entities: A, B, a, b, ...
    #   - Hypernym-hyponym relations: A/B is the hyponym of a/b, i.e., a and b are more specific concepts to A and B, respectively.
    #   - Triples after filtering: (a, has_result, b), ...
    #   - Entities after filtering: A, B, a, b, ...
    for doc_i in tqdm(range(len(documents))):
        document = documents[doc_i]
        triples = document["triples"] # We check these tripls
        # Get tree-number-based triples for EACH triple
        all_tree_number_triples = []
        for triple in triples:
            head_entity_id = triple["head_entity"]["id"]
            tail_entity_id = triple["tail_entity"]["id"]
            relation = triple["relation"]["name"]
            lst = [] # tree-number-based triples for `triple`
            for head_tree_number in entity_id_to_tree_numbers[head_entity_id]:
                for tail_tree_number in entity_id_to_tree_numbers[
                    tail_entity_id
                ]:
                    lst.append((head_tree_number, relation, tail_tree_number))
            all_tree_number_triples.append(lst)
        # Give hypernym marks
        hypernym_count = 0
        for triple1_i in range(len(triples)):
            # triple1 = triples[triple1_i]
            # Check whether this `triple1` is a hypernym for other triples
            # Compare the tree-number-based triples for `triple1` with the tree-number-based triples of others
            tree_number_triples1 = all_tree_number_triples[triple1_i]
            hypernym = False
            hyponym_triple_i = None
            for triple2_i in range(len(triples)):
                if triple1_i == triple2_i:
                    continue
                tree_number_triples2 = all_tree_number_triples[triple2_i]
                if is_hypernym_triple(
                    tree_number_triples1=tree_number_triples1,
                    tree_number_triples2=tree_number_triples2
                ):
                    hypernym = True
                    hyponym_triple_i = triple2_i
                    hypernym_count += 1
                    break
            triples[triple1_i]["is_hypernym"] = hypernym
            if hypernym:
                hyponym_triple = triples[hyponym_triple_i]
                hyponym_triple = (
                    hyponym_triple["head_entity"]["name"],
                    hyponym_triple["relation"]["name"],
                    hyponym_triple["tail_entity"]["name"]
                )
                triples[triple1_i]["hyponym_triple"] = hyponym_triple
        document["triples"] = triples
        if hypernym_count > 0:
            print(f"Found {hypernym_count} hypernym triples (over {len(triples)} triples) for {doc_i}-th document")
            assert hypernym_count < len(triples)
    return documents


def is_hypernym_triple(tree_number_triples1, tree_number_triples2):
    # If tree number A is completely contained in tree number B,
    # A is a hypernym entity of B (i.e., B is more specific).
    for h1, r1, t1 in tree_number_triples1:
        for h2, r2, t2 in tree_number_triples2:
            if (h1 == h2) and (t1 == t2):
                continue
            if (h1 in h2) and (r1 == r2) and (t1 in t2):
                print(f"Hypernym: {(h1,t1,r1)}")
                print(f"Hyponym:  {(h2,t2,r2)}")
                return True
    return False


# ---------------------------


# def transform_to_kapipe_format(documents):
#     results = [] # list[Document]
# 
#     for document in documents:
#         result = {}
# 
#         result["doc_key"] = document["doc_key"]
#         result["pubmed_id"] = document["pubmed_id"]
#         result["sentences"] = document["sentences"]
# 
#         # First, collect course-grained/fine-grained entities and their mapping
#         entity_ids = []
#         entity_id_to_names = defaultdict(list)
#         entity_id_to_types = defaultdict(list)
#         fg_entity_ids = []
#         fg_entity_id_to_names = defaultdict(list)
#         fg_entity_id_to_types = defaultdict(list)
#         fine_to_coarse_map = defaultdict(list)
#         for triple in document["triples"]:
#             # head
#             entity_id, entity_name, entity_type = get_id_name_type(
#                 triple=triple,
#                 key="head_entity"
#             )
#             entity_ids.append(entity_id)
#             entity_id_to_names[entity_id].append(entity_name)
#             entity_id_to_types[entity_id].append(entity_type)
#             # fine-grained head
#             fg_entity_id, fg_entity_name, fg_entity_type = get_id_name_type(
#                 triple=triple,
#                 key="fg_head_entity"
#             )
#             fg_entity_ids.append(fg_entity_id)
#             fg_entity_id_to_names[fg_entity_id].append(fg_entity_name)
#             fg_entity_id_to_types[fg_entity_id].append(fg_entity_type)
# 
#             fine_to_coarse_map[fg_entity_id].append(entity_id)
# 
#             # tail
#             entity_id, entity_name, entity_type = get_id_name_type(
#                 triple=triple,
#                 key="tail_entity"
#             )
#             entity_ids.append(entity_id)
#             entity_id_to_names[entity_id].append(entity_name)
#             entity_id_to_types[entity_id].append(entity_type)
#             # fine-grained tail
#             fg_entity_id, fg_entity_name, fg_entity_type = get_id_name_type(
#                 triple=triple,
#                 key="fg_tail_entity"
#             )
#             fg_entity_ids.append(fg_entity_id)
#             fg_entity_id_to_names[fg_entity_id].append(fg_entity_name)
#             fg_entity_id_to_types[fg_entity_id].append(fg_entity_type)
# 
#             fine_to_coarse_map[fg_entity_id].append(entity_id)
# 
#         entity_ids = sorted(list(set(entity_ids)))
#         for e_id in entity_ids:
#             assert len(set(entity_id_to_names[e_id])) == 1
#             assert len(set(entity_id_to_types[e_id])) == 1
#         fg_entity_ids = sorted(list(set(fg_entity_ids)))
#         for e_id in fg_entity_ids:
#             assert len(set(fg_entity_id_to_names[e_id])) == 1
#             assert len(set(fg_entity_id_to_types[e_id])) == 1
#             assert len(set(fine_to_coarse_map[e_id])) == 1
# 
#         # Set mentions
#         # result["mentions"] = None
# 
#         # Set entities
#         entities = []
#         entity_id_to_index = {}
#         for e_i, e_id in enumerate(entity_ids):
#             entity = {
#                 # "mention_indices": None,
#                 "entity_id": e_id,
#                 "name": entity_id_to_names[e_id][0],
#                 "entity_type": entity_id_to_types[e_id][0],
#             }
#             entities.append(entity)
#             entity_id_to_index[e_id] = e_i
#         result["entities"] = entities
# 
#         # Set fine-grained entities
#         fg_entities = []
#         fg_entity_id_to_index = {}
#         for e_i, e_id in enumerate(fg_entity_ids):
#             coarse_ent_idx = entity_id_to_index[fine_to_coarse_map[e_id][0]]
#             entity = {
#                 # "mention_indices": None,
#                 "entity_id": e_id,
#                 "name": fg_entity_id_to_names[e_id][0],
#                 "entity_type": fg_entity_id_to_types[e_id][0],
#                 "coarse_grained_entity_index": coarse_ent_idx,
#             }
#             fg_entities.append(entity)
#             fg_entity_id_to_index[e_id] = e_i
#         result["fine_grained_entities"] = fg_entities
# 
#         # Set relations
#         relations = []
#         for triple in document["triples"]:
#             head_entity_id = triple["head_entity"]["id"]
#             head_entity_idx = entity_id_to_index[head_entity_id]
#             tail_entity_id = triple["tail_entity"]["id"]
#             tail_entity_idx = entity_id_to_index[tail_entity_id]
#             relation = triple["relation"]["name"]
#             relations.append((head_entity_idx, relation, tail_entity_idx))
#         relations = sorted(list(set(relations)), key=lambda x: (x[0],x[2],x[1]))
#         relations = [
#             {
#                 "arg1": triple[0],
#                 "relation": triple[1],
#                 "arg2": triple[2]
#             }
#             for triple in relations]
#         result["relations"] = relations
# 
#         # Set fine-grained relations
#         fg_relations = []
#         for triple in document["triples"]:
#             fg_head_entity_id = triple["fg_head_entity"]["id"]
#             fg_head_entity_idx = fg_entity_id_to_index[fg_head_entity_id]
#             fg_tail_entity_id = triple["fg_tail_entity"]["id"]
#             fg_tail_entity_idx = fg_entity_id_to_index[fg_tail_entity_id]
#             relation = triple["relation"]["name"]
#             fg_relations.append((fg_head_entity_idx, relation, fg_tail_entity_idx))
#         fg_relations = sorted(list(set(fg_relations)), key=lambda x: (x[0],x[2],x[1]))
#         fg_relations = [
#             {
#                 "arg1": triple[0],
#                 "relation": triple[1],
#                 "arg2": triple[2]
#             }
#             for triple in fg_relations]
#         result["fine_grained_relations"] = fg_relations
# 
#         results.append(result)
# 
#     return results
# 
# 
# def get_id_name_type(triple, key):
#     return triple[key]["id"], triple[key]["name"], triple[key]["type"]
# 

# ---------------------------


def split_documents(documents, n_dev, n_test):
    """Split documents into training, development, and test sets.

    Parameters
    ----------
    documents : list[Doc]
    n_dev : int
    n_test : int

    Returns
    -------
    list[Doc], list[Doc], list[Doc]
    """
    # Split documents based on pubmed ID

    # We first create a mapping from Pubmed ID to doc_keys
    pubmed_id_to_doc_indices = defaultdict(list)
    for doc_i in range(len(documents)):
        document = documents[doc_i]
        pubmed_id = document["pubmed_id"]
        pubmed_id_to_doc_indices[pubmed_id].append(doc_i)

    test_doc_indices = []
    dev_doc_indices = []
    train_doc_indices = []
    pubmed_id_list = list(pubmed_id_to_doc_indices.keys())
    np.random.seed(12345)
    np.random.shuffle(pubmed_id_list)
    idx = 0
    while len(test_doc_indices) < n_test:
        pubmed_id = pubmed_id_list[idx]
        doc_indices = pubmed_id_to_doc_indices[pubmed_id]
        test_doc_indices.extend(doc_indices)
        idx += 1
    while len(dev_doc_indices) < n_dev:
        pubmed_id = pubmed_id_list[idx]
        doc_indices = pubmed_id_to_doc_indices[pubmed_id]
        dev_doc_indices.extend(doc_indices)
        idx += 1
    for i in range(idx, len(pubmed_id_list)):
        pubmed_id = pubmed_id_list[i]
        doc_indices = pubmed_id_to_doc_indices[pubmed_id]
        train_doc_indices.extend(doc_indices)

    assert len(train_doc_indices) + len(dev_doc_indices) + len(test_doc_indices) == len(documents)

    train_documents = create_subset(
        documents=documents,
        doc_indices=train_doc_indices
    )
    dev_documents = create_subset(
        documents=documents,
        doc_indices=dev_doc_indices
    )
    test_documents = create_subset(
        documents=documents,
        doc_indices=test_doc_indices
    )

    print(f"Number of training documents: {len(train_documents)}")
    print(f"Number of development documents: {len(dev_documents)}")
    print(f"Number of test documents: {len(test_documents)}")
    return train_documents, dev_documents, test_documents


def create_subset(documents, doc_indices):
    subset = []
    for doc_i in doc_indices:
        subset.append(documents[doc_i])
    return subset


# ---------------------------


def read_csv(path, delimiter, with_head, with_id, encoding="utf-8"):
    header = 0 if with_head else None
    index_col = 0 if with_id else None
    data = pd.read_csv(path, encoding=encoding, delimiter=delimiter, header=header, index_col=index_col)
    return data


def read_json(path, encoding=None):
    if encoding is None:
        with open(path) as f:
            dct = json.load(f)
    else:
        with io.open(path, "rt", encoding=encoding) as f:
            line = f.read()
            dct = json.loads(line)
    return dct


def write_json(path, dct):
    with open(path, "w") as f:
        # json.dump(dct, f, indent=4)
        json.dump(dct, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--ontology", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    main(args=args)
