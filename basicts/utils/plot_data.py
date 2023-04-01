import matplotlib as plt
def plot_data( pre, real, selected_node_id=0, line_width: float = 1.5, font_size: int = 16, history_color="blue", real_color="red", figure_size: tuple = (10, 5)):
    num_samples = pre.shape[1]
    plt.rcParams['figure.figsize'] = figure_size
    plt.plot(real[:,0, selected_node_id].squeeze(),linewidth=line_width, color=real_color, label="Ground Truth")
    plt.plot(pre[:,0, selected_node_id].squeeze(),linewidth=line_width, color=history_color, label=self.exp_name)
    plt.grid()
    plt.legend(fontsize=font_size)
    plt.savefig('prediction_{0}.png'.format(self.exp_name), dpi=500)
    plt.show()
    plt.clf()