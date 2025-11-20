import streamlit as st
import json
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt


st.title("ISLR Confusion Matrix Analysis")

st.sidebar.header("Experiment Configuration")

size = st.sidebar.selectbox("Size of dataset", [20000, 40000])
train_signs = st.sidebar.selectbox("Number of train signs", [200, 300, 400, 500])
test_signs = 200
st.sidebar.write("Number of test signs: 200")
seed = st.sidebar.selectbox("Seed", [42, 44, 46])

confusion_path = os.path.join(os.path.curdir, "confusions")
data_path = os.path.join(os.path.curdir, "data_resplit_v2")

experiment_path = f'size_{size}/test_{test_signs}/train_{train_signs}/seed_{seed}'
metadata_path = os.path.join(data_path, experiment_path, "metadata.json")
target_cm_path = os.path.join(confusion_path, experiment_path, f"cm_size{size}_train{train_signs}_test{test_signs}_seed{seed}_fc-test-subset.csv")
all_cm_path = os.path.join(confusion_path, experiment_path, f"cm_size{size}_train{train_signs}_test{test_signs}_seed{seed}_fc-train.csv")
test_data_path = os.path.join(data_path, experiment_path, "test.csv")

with open(metadata_path, "r") as f:
  metadata = json.load(f)

tab1, tab2 = st.tabs(["Confusion Matrix", "Homonym Analysis"])

with tab1:
    st.subheader("Experiment Details")
    st.write(f'SIze: {size}, Number of test signs: {test_signs}, Number of train signs: {train_signs}, Seed: {seed}')

    test_signs = metadata.get("test_signs", [])
    train_signs = metadata.get("train_signs", [])
    additional_train_signs = list(set(train_signs) - set(test_signs))
    st.write("Number of additional train signs: " + str(len(additional_train_signs)))

    st.subheader("Confusion Matrix")
    # Read confusion matrix for classification head on all signs in testing (all train signs) set
    all_cm_df = pd.read_csv(all_cm_path)
    train_idx_to_label = dict(enumerate(sorted(metadata['train_signs'])))
    all_cm_df['true_label'] = all_cm_df['true_idx'].map(train_idx_to_label)
    all_cm_df['pred_label'] = all_cm_df['pred_idx'].map(train_idx_to_label)

    # Read confusion matrix for classification head on signs only in target (test_signs) set
    cm_df = pd.read_csv(target_cm_path)
    test_idx_to_label = dict(enumerate(sorted(metadata['test_signs'])))
    cm_df['true_label'] = cm_df['true_idx'].map(test_idx_to_label)
    cm_df['pred_label'] = cm_df['pred_idx'].map(test_idx_to_label)

    classes = sorted(set(cm_df.true_label).union(cm_df.pred_label))

    pivot_cm = cm_df.pivot_table(
        index="true_label",
        columns="pred_label",
        values="count",
        aggfunc="sum",
        fill_value=0
    ).reindex(index=classes, columns=classes, fill_value=0)

    st.write(pivot_cm)

    diagonal_values = pivot_cm.values.diagonal().sum()
    all_values = pivot_cm.values.sum()
    accuracy = diagonal_values / all_values
    st.metric("Accuracy", f"{accuracy * 100:.2f}%")

    confusions = []
    for true_label in pivot_cm.index:
        for pred_label in pivot_cm.columns:
            if true_label != pred_label and pivot_cm.loc[true_label, pred_label] > 0:
                confusions.append((pivot_cm.loc[true_label, pred_label], true_label, pred_label))

    confusions.sort(reverse=True)

    st.subheader("Top Confusions")
    for i in range(5):
        pair = confusions[i]
        st.write("Top " + str(i+1) + " Confusion - True label: " + pair[1] + ", Predicted label: " + pair[2] + ", Count: " + str(pair[0]))

with tab2:
    homosigns = [
        ["alot", "much"],
        ["ant", "bug"],
        ["around", "turnaround"],
        ["asleep", "sleep"],
        ["bathtub", "bath"],
        ["bear", "teddy"],
        ["big", "tall"],
        ["break", "broken"],
        ["cartcarriage", "hammer"],
        ["chicken", "bird"],
        ["coats", "jacket"],
        ["cook", "kitchen", "share"],
        ["couch", "bench", "porch", "sofa"],
        ["could", "can"],
        ["deer", "moose"],
        ["dont", "not"],
        ["dress", "shirt"],
        ["eat", "food"],
        ["enter", "into", "slipper"],
        ["gentle", "soft", "wet"],
        ["get", "tights"],
        ["goingto", "gotto", "go"],
        ["hand", "wait", "finger"],
        ["hug", "love"],
        ["little", "short", "child"],
        ["must", "haveto"],
        ["needneedto", "should"],
        ["nightnight", "tonightnight"],
        ["other", "another"],
        ["over", "after"],
        ["party", "play", "yellow"],
        ["dump", "pour"],
        ["present", "gift"],
        ["bunny", "rabbit"],
        ["rock", "stone"],
        ["slide", "slideverb"],
        ["sneaker", "shoe"],
        ["swim", "pool"],
        ["taken", "fast"],
        ["toilet", "bathroom"],
        ["was", "were"],
        ["wind", "windy"],
        ["wish", "hungry"],
        ["awake", "wake"],
        ["glasswindow", "tooth"],
        ["beside", "person"],
        ["pen", "pencil"],
        ["hi", "hello"]
    ]

    st.subheader("Known homosigns:")
    st.write(homosigns)

    homosigns_in_test = []
    homosigns_and_corresponding_in_test = []
    additional_homonyms_in_train = []

    # goes through all homosigns
    for homosign in homosigns:
        for sign in homosign:
            # if a homosign appears in test set
            if sign in set(test_signs):
                homosigns_in_test.append(sign)
                if len(homosign) > 1:
                    others = homosign[:]
                    others.remove(sign)
                    for other in others:
                        if other in set(test_signs):
                            homosigns_and_corresponding_in_test.append((sign, other))

                # check if corresponding homosigns appear in additional train set
                for other_sign in homosign:
                    if other_sign in set(additional_train_signs):
                        additional_homonyms_in_train.append((sign, other_sign))

    st.subheader("Homosigns in test & train:")
    st.write(str(len(homosigns_in_test)) + " homonymns present in test set:")
    st.write(sorted(homosigns_in_test))

    st.write(str(len(additional_homonyms_in_train)) + " corresponding homonyms present in additional train set (homonym_in_test, corresponding)")
    st.write(sorted(additional_homonyms_in_train))

    # merge mapping
    mapping = {}
    for pair in homosigns_and_corresponding_in_test:
        if pair[0] not in mapping:
            mapping[pair[0]] = pair[0] + "_" + pair[1]
        if pair[1] not in mapping:
            mapping[pair[1]] = pair[0] + "_" + pair[1]


    merged_cm = (
        pivot_cm
        .rename(index=mapping, columns=mapping)
        .groupby(level=0, axis=0).sum()
        .groupby(level=0, axis=1).sum()
    )

    st.subheader("Merged Confusion Matrix")
    st.write(merged_cm)

    merged_diagonal_values = merged_cm.values.diagonal().sum()
    merged_all_values = merged_cm.values.sum()
    merged_accuracy = merged_diagonal_values / merged_all_values
    st.metric("Accuracy", f"{merged_accuracy * 100:.2f}%")

    merged_confusions = []
    for true_label in merged_cm.index:
        for pred_label in merged_cm.columns:
            if true_label != pred_label and merged_cm.loc[true_label, pred_label] > 0:
                merged_confusions.append((merged_cm.loc[true_label, pred_label], true_label, pred_label))

    merged_confusions.sort(reverse=True)

    st.subheader("Top confusions - Merged CM")
    for i in range(5):
        pair = merged_confusions[i]
        st.write("Top " + str(i+1) + " Confusion - True label: " + pair[1] + ", Predicted label: " + pair[2] + ", Count: " + str(pair[0]))