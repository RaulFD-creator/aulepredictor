"""
Functions for registering protein regions into the different training/validation folds.

Copyright by Raúl Fernández Díaz
"""

import os
import random
import pandas as pd

from sklearn.model_selection import train_test_split, KFold

def register_data(output_dir : str, database : str, metal : str, n_splits : int) -> None:
    """
    Register the protein regions into different training/validation folds,
    ensuring that 1) regions belonging to the same protein have to be in 
    the same fold, 2) a proportion close to 1:1 between metal-binding and
    non-binding regions must be maintained, and 3) there can only be
    20 regions from a single protein.

    Parameters
    ----------

    
    """
    k = 0
    structures = []
    Info = []
    names = []

    for file in os.listdir(os.path.join(output_dir, "metal_binding")):
        structures.append(_create_entry(file, os.path.join(output_dir, "metal_binding"), metal, 1, k))
        current_name = file.split("_")[0]
        if current_name not in names:
            Info.append(_create_super_entry(file.split("_")[0], 1))
            names.append(current_name)
        k += 1

    for file in os.listdir(os.path.join(output_dir, "not_binding")):
        structures.append(_create_entry(file, os.path.join(output_dir, "not_binding"), metal, 0, k))
        current_name = file.split("_")[0]
        if current_name not in names:
            Info.append(_create_super_entry(file.split("_")[0], 0))
            names.append(current_name)
        k += 1

    k = 0
    random.shuffle(Info)
    df = pd.DataFrame(Info)
    random.shuffle(structures)
    df_struct = pd.DataFrame(structures)

    for protein in df_struct['protein_ID'].unique():
        k += 1
        print(f"\nProteins analysed: {k}")
        protein_counts = len(df_struct[df_struct['protein_ID']==protein])
        a = df_struct[df_struct['protein_ID']==protein]
        if protein_counts > 20:
            difference = protein_counts - 20
            counter = 0
            to_remove = []
            print(f"To be removed: {difference}")
            for idx in a.index:
                if counter < difference:
                    to_remove.append(idx)
                    counter += 1
                else:
                    break
            df_struct.drop(to_remove, axis=0, inplace=True)

        else:
            continue

    counter = 0
    k = 0

    for protein in df_struct['protein_ID'].unique():
        k += 1
        print(f"\nProteins analysed: {k}")
        try:
            majority = df_struct[df_struct['protein_ID']==protein]["Binding"].value_counts()[0]
            minority = df_struct[df_struct['protein_ID']==protein]["Binding"].value_counts()[1]
            if majority < minority: majority, minority = minority, majority
        except KeyError:
            continue
        difference = majority - minority
        a = df_struct[df_struct['protein_ID']==protein]
        if difference > 0:
            counter = 0
            to_remove = []
            print(f"To be removed: {difference}")
            for idx in a.index:
                if counter < difference:
                    to_remove.append(idx)
                    counter += 1
                else:
                    break
            df_struct.drop(to_remove, axis=0, inplace=True)

        else:
            continue

    print(f"Non-binding {df_struct['Binding'].value_counts()[0]}")
    print(f"Metal-binding {df_struct['Binding'].value_counts()[1]}")


    print("\nCreating training/testing split")
    x_train, x_test, y_train, y_test = train_test_split(df, df["binding"], test_size=0.1, random_state=2812)
    _create_csv(x_test, f"{database}_testing", df_struct)
    del(x_test)
    del(y_test)

    print("Creating Kfolds")
    KF = KFold(n_splits=n_splits)
    k = 0
    for train_index, val_index in KF.split(x_train):
        print(f"Creating fold: {k}")
        _create_csv(x_train.iloc[train_index], f"{database}_fold_train_{k}", df_struct)
        _create_csv(x_train.iloc[val_index], f"{database}_fold_val_{k}", df_struct)
        k += 1


def _create_entry(file, output_dir, metal, binding, entry_no):
    """
    Function to create entries in a csv, where the information regarding each minibox will
    be recorded. The entries will be passed as dictionaries. The information will be:

    - Entry: entry number
    - Path: whole path of the minibox
    - Filename: identificative name of the minibox
    - Metal: which metal is the one bound to the minibox
    - Binding: whether the minibox contains a metal center (1) or not (0)    
    """

    filename = file.split(".")[0]
    path = os.path.join(output_dir, "metal_binding" if binding == 1 else "not_binding", filename + ".pt")
    entry = {"Entry": entry_no,
            "protein_ID": file.split("_")[0],
            "Filename": filename,
            "Path": path, 
            "Metal": metal,
            "Binding": binding}
    return entry

def _create_super_entry(name, binding):
    entry = {
        "name": name,
        "binding": binding
    }
    return entry

def _create_csv(data, dataset, df):
    concatenate = []
    names = list(data["name"])

    for _, structure in df.iterrows():
        if structure["protein_ID"] in names:
            concatenate.append(structure)
    concatenate = pd.DataFrame(concatenate)
    concatenate.to_csv("./tmp.csv")

    with open("./tmp.csv") as fi:
        fi.readline()
        if not os.path.exists(os.path.join(f".", f"{dataset}.csv")):
            with open(os.path.join(f'.', f'{dataset}.csv'),"w") as fo:
                fo.write("Path,Binding\n")
        if not os.path.exists(os.path.join(f".", "Complete_registry.csv")):
                with open(os.path.join(f".", "Complete_registry.csv"), "w") as fo2:
                    fo2.write("Protein_ID,Path,Binding,Metal,Fold\n")
        with open(os.path.join(f".", f"{dataset}.csv"), "a") as fo:
            with open(os.path.join(f".", "Complete_registry.csv"), "a") as fo2:
                for line in fi:
                    info = line.split(",")
                    new_content = info[4] + "," + info[6]
                    fo.write(new_content)
                    new_content = info[2]  + "," + info[4] + "," + info[6].strip("\n") + "," +  info[5] + "," + f"{dataset}" + "\n"
                    fo2.write(new_content)
