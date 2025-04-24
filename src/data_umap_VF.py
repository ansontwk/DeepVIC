import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import umap
import time
from utils.plotting import color_dict, colour_handle, colour_handle_nouncls, plot_umap, filter_unclassified as filter_uncls, filter_VF as filter 
from utils.formatting import multimodal
start_time = time.time()

def main():
    tensor_concat, vf, cls = multimodal()
    pos_tensor_nouncls, pos_cls_nouncls = filter_uncls(tensor_concat, vf, cls)
    pos_tensor, pos_cls = filter(tensor_concat, vf, cls)
    color = [color_dict[label] for label in pos_cls]
    color_nouncls = [color_dict[label] for label in pos_cls_nouncls]
    
    reducer = umap.UMAP(random_state=42, 
                        low_memory=False, 
                        n_neighbors=50, 
                        min_dist=0.5)
    
    embedding = reducer.fit_transform(pos_tensor)
    embedding_nouncls = reducer.fit_transform(pos_tensor_nouncls)

    plot_umap(embedding_nouncls, color_nouncls, colour_handle_nouncls, f"./plot/UMAP_VF_uncls.pdf")
    plot_umap(embedding, color, colour_handle, f"./plot/UMAP_VF.pdf", col_alpha = 0.8)
main()
end_time = time.time()
print(f"Total time elapsed: {end_time - start_time} seconds")
