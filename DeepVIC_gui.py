import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from transformers.utils import logging
logging.set_verbosity(40)
import gradio as gr
import pandas as pd
import numpy as np
from tensorflow import keras
from Bio import SeqIO
from src.utils.headers import version, program_desc
from src.utils.paths import model_binary, ProtBertBFD_path, model_multiclass
from src.utils.predict_utils import load_data_seqonly as load_data, embed_PLM as embed, bin2VF, get_PSSM_embeddings, PSSM_TYPE as pssm_dict, ORDERED_OPT_PSSM as features, mult2VF
THEME = gr.themes.Glass()

def get_fastaname(fastafile):
    gr.Info(f"Uploaded {os.path.basename(fastafile)} successfully")
    
def preview_sequences(fastafile):
    gr.Info(f"Previewing {os.path.basename(fastafile)}")
    if not fastafile:
        return "Please upload a valid FASTA file"
    headers = []
    with open(fastafile, "r") as readfile:
        for record in SeqIO.parse(readfile, "fasta"):
            headers.append(str(record.id))
    return "\n".join(headers)

def binary(fastafile, bertpath):
    gr.Info(f"Running binary prediction on {os.path.basename(fastafile)}")
    if not fastafile:
        return pd.DataFrame({"Headers": ["Please upload a valid FASTA file"], "Predictions": ['na']})
    seqs, headers = load_data(fastafile)
    if bertpath:
        embeddings = embed(seqs, bertpath)
    else:
        embeddings = embed(seqs, ProtBertBFD_path)
    model = keras.models.load_model(model_binary)
    bin_pred = model.predict(embeddings, verbose = 0)
    pred_bin_idx = bin2VF(bin_pred)
    df = pd.DataFrame({'Headers': headers, 'Predictions': pred_bin_idx})
    return df

def multiclass(fastafile, bertpath, pssmpath):
    gr.Info(f"Running multiclass prediction on {os.path.basename(fastafile)}")
    if not fastafile:
        return pd.DataFrame({"Headers": ["Please upload a valid FASTA file"], "Predictions": ['na']})
    seqs, headers = load_data(fastafile)
    
    if bertpath:
        embeddings = embed(seqs, bertpath)
    else:
        embeddings = embed(seqs, ProtBertBFD_path)

    if pssmpath:
        pssm_embeddings = get_PSSM_embeddings(headers, features, pssm_dict, pssmpath)
    else:
        pssm_embeddings = get_PSSM_embeddings(headers, features, pssm_dict)
        
    tensor_concat = np.concatenate((embeddings, pssm_embeddings), axis=1)
    model = keras.models.load_model(model_multiclass)
    mult_pred = model.predict(tensor_concat, verbose = 0)
    pred_mult_idx = mult2VF(mult_pred)
    df = pd.DataFrame({'Headers': headers, 'Predictions': pred_mult_idx})
    return df
    
def save_file(df, savepath):
    if savepath:
        file_path = os.path.join(savepath, "predictions.tsv")
    else:
        file_path = "./output/predictions.tsv"
    gr.Info(f"Saving predictions to {file_path}")
    df.to_csv(file_path, index=False, sep = "\t")
    return file_path

def debugger(protbert, outpath):
    gr.Info("Running debugger")
    if not protbert:
        protbert = ProtBertBFD_path
    if not outpath:
        outpath = "./output/predictions.tsv"
    order = ["Protbert:", "Output Path:", "Binary Model:", "Multi-class Model:"]
    outputtext = []
    for tag, value in zip(order, [protbert, outpath, model_binary, model_multiclass]):
        outputtext.append(f"{tag} {value}")
    output = "\n".join(outputtext)
    return output

def main():
    with gr.Blocks(theme = THEME) as deepvic:    
        with gr.Column():
            gr.Markdown(f"# {program_desc}")
            gr.Markdown(f"## Version {version}")
            gr.Markdown("### You are running the GUI version of DeepVIC. For more customization and control over pipeline parameters, please use the command line version.")

            inputfile = gr.File(label = "Place your protein fasta file here", file_types=[".fasta", ".faa", ".fa"])
            protbert_bfd_pathbox = gr.Textbox(label="Path to ProtBERT-BFD model.", info = f"Leave blank for the default value of {ProtBertBFD_path} as defined in src/utils/paths.py")
                
            with gr.Accordion("Multiclass model parameters", open = True):
                gr.Markdown("### For multiclass prediction, please provide the following parameters")
                pssm_pathbox = gr.Textbox(label=f"Path to PSSM results.", info = "Your sequence may not yield PSSM results. Leave blank in that case.")
                    
                
            with gr.Accordion("Preview Sequences", open = False): #gr.Column():
                preview_box = gr.Textbox(label="Sequence Headers Preview", info = "Headers will be displayed here", interactive=False)
                preview_buttom = gr.Button("Preview Sequences")
                    
            #========================================================
            #prediction button
            with gr.Row():
                predict_button = gr.Button("Predict Binary model")
                predict_mult_button = gr.Button("Predict Multiclass Results")
                
            #========================================================
            #prediction results section

            output_df = gr.DataFrame(label="Prediction Results")
            save_path_box = gr.Textbox(label="Save Path", info= "Leave blank to save into the 'output' folder of DeepVIC.")
            save_button = gr.Button("Save Predictions")
            output_file = gr.File(interactive = False, visible = False)
            with gr.Accordion("Debugger", open = False):
                debugger_textbox = gr.Textbox(label="Debugger", interactive=False)
                debugger_button = gr.Button("Show paths and models")

        #actions
        inputfile.change(get_fastaname, inputfile, [])
        preview_buttom.click(preview_sequences, inputfile, preview_box)
        predict_button.click(binary, [inputfile, protbert_bfd_pathbox], output_df)
        predict_mult_button.click(multiclass, [inputfile, protbert_bfd_pathbox, pssm_pathbox], output_df)
        save_button.click(save_file, [output_df, save_path_box], output_file)
        debugger_button.click(debugger, [protbert_bfd_pathbox, save_path_box], debugger_textbox)
    deepvic.launch()

main()