AMINO = {
    'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5,
    'H': 6, 'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11,
    'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17,
    'W': 18, 'Y': 19, 'X': 20, '*': 21
}
AMINOLIST = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]
FEATURES = ["aac_pssm", "ab_pssm", "d_fpssm", "dpc_pssm",
            "edp", "eedp", "pssm_composition", "rpm_pssm",
            "rpssm", "s_fpssm", "tpc", "k_separated_bigrams_pssm", "pse_pssm"]
OPT_PSSM = ['aac_pssm', 'd_fpssm', 'edp', 'pssm_composition', 'rpm_pssm', 'k_separated_bigrams_pssm']
ORDERED_OPT_PSSM = ["aac_pssm", "d_fpssm", "edp", "k_separated_bigrams_pssm", "pssm_composition", "rpm_pssm"]

PSSM_TYPE = {"aac_pssm": {"default": 20,
                             "embeddings" : []},
                "ab_pssm" : {"default" : 400,
                              "embeddings" : []}, 
                "d_fpssm" : {"default": 20,
                              "embeddings" : []},
                "dpc_pssm" : {"default" : 400,
                               "embeddings" : []}, 
                "edp" : {"default" : 20,
                          "embeddings" : []}, 
                "eedp" : {"default" : 400,
                          "embeddings" : []}, 
                "k_separated_bigrams_pssm" : {"default" : 400,
                                             "embeddings" : []},
                "pse_pssm" : {"default" : 40,
                              "embeddings" : []},
                "pssm_composition" : {"default" : 400,
                                      "embeddings" : []}, 
                "rpm_pssm" : {"default" : 400,
                               "embeddings" : []}, 
                "rpssm" : {"default" : 110,
                            "embeddings" : []},
                "s_fpssm" : {"default" : 400,
                              "embeddings" : []},
                "tpc" : {"default" : 400,
                         "embeddings" : []}
                }
idx_to_VF = {
    0: "Adherence",
    1: "Invasion",
    2: "Effector Delivery System",
    3: "Motility",
    4: "Exotoxin",
    5: "Exoenzyme",
    6: "Immune modulation",
    7: "Biofilm",
    8: "Nutritional/metabolic factor",
    9: "Stress survival",
    10: "Regulation",
    11: "Post-translational modification",
    12: "Antimicrobial activity/competitive advantage",
    13: "Others"
}
