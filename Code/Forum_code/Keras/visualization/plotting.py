import pandas
from matplotlib import pyplot as plt
#%matplotlib inline
plt.style.use('ggplot')

def plot_res(columns, ylab):
    
    p = df[columns].plot(figsize=(15,10))
    p.set_xlabel("Iterations")
    p.set_ylabel(ylab)
    p.set_title(ylab + " change during training")
    p.axes.get_xaxis().set_visible(True)


def write_images(folder):

    filename = folder + '/RMSE_CRPS.txt'
    df = pandas.read_table(filename)

    plot_res(["train_RMSE_d", "test_RMSE_d"], "loss")
    plt.savefig(folder + '/diastolic_loss.png')

    plot_res(["train_RMSE_s", "test_RMSE_s"], "loss")
    plt.savefig(folder + '/systolic_loss.png')

    plot_res(["train_crps", "test_crps"], "CRPS")
    plt.savefig(folder + '/crps.png')
