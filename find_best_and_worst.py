# Third Party Packages
import matplotlib.pyplot as plt
import numpy as np
# Build-in Packages
import os
from heapq import heappop, heappush
# Local Packages
from neural_network.neural_network import model_loader
from neural_network.data_loader import mnist_loader

def plot_result(test_set, result_list, fig_title):
    num_row = 2
    num_col = len(result_list)//num_row
    # plot images
    fig, axes = plt.subplots(num_row, num_col, figsize=(2*num_col,2*num_row))

    fig.suptitle(fig_title, fontsize=12)
    for i in range(len(result_list)):
        cost, img_idx, prediction, confidence, _ = result_list[i]
        ax = axes[i//num_col, i%num_col]
        image = np.reshape(test_set[img_idx][0],(28,28))
        ax.imshow(image, cmap='gray')
        label = ("Prediction: " + str(prediction) + 
                "\nCost: " + "{:.2e}".format(cost) +
                "\nConfidence: " + str(confidence*100)[:4] + "%"
                )
        ax.set_title(label)
    plt.tight_layout()
    plt.show()

output_file_path = os.path.join(os.path.dirname(__file__),'result\model_pkl')
training_set, validation_set, test_set = mnist_loader()
net2 = model_loader(output_file_path)
evaluations = net2.evaluate_testing_data(test_set)
best = []
number_of_sample = 6
worst = []
for cost, id, predict, conf, res in evaluations:
    heappush(best, (-cost,id,predict,conf,res))
    if len(best) >number_of_sample:
        heappop(best)
    heappush(worst,(cost,id,predict,conf,res))
    if len(worst) >number_of_sample:
        heappop(worst)


structure = net2.get_detail()["structure"]
active_method = net2.get_detail()["activation"]
network_detail =  f"Structure: {structure} \nActivation Function: {active_method}"
plot_result(test_set,worst, f"{network_detail}\nWorst {number_of_sample}")
plot_result(test_set,best, f"{network_detail}\nBest {number_of_sample}")



