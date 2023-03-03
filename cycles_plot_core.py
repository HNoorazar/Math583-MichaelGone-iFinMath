import matplotlib.pyplot as plt


def plot_portraits(images, titles, h, w, n_row, n_col):
    """
    It helps visualising the portraits from the dataset.
    """

    plt.figure(figsize=(2.2 * n_col, 2.2 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.20)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i])
        plt.xticks(())
        plt.yticks(())




class two_curves_2subplots_class:
  def __init__(self, x1, y1, leg1, x2, y2, leg2, x_label, y_label, x_limits, y_limits):
    self.x1 = x1
    self.y1 = y1
    self.leg1 = leg1

    self.x2 = x2
    self.y2 = y2
    self.leg2 = leg2
    
    self.x_label = x_label
    self.y_label = y_label
    self.x_limits = x_limits
    self.y_limits = y_limits

def plot_TwoSubplots_Vertical_oneCurvePerSub(two_subplots_obj):
    # plt.subplots_adjust(left=0, bottom=0, right=1.1, top=0.9, wspace=0, hspace=0)
    # plt.subplots_adjust(wspace=.1, hspace = 0.5)
    plt.subplots_adjust(right=1.5, top=1.1)

    ##########################################################################################
    fig, axs = plt.subplots(2, 1, figsize=(10, 6),
                            sharex='col', sharey='row',
                            gridspec_kw={'hspace': 0.2, 'wspace': .1});

    (ax1, ax2) = axs;
    ax1.grid(True); ax2.grid(True)


    #######
    #######  subplot 1
    #######
    ax1.plot(two_subplots_obj.x1, two_subplots_obj.y1, '-', label = two_subplots_obj.leg1, c='r') #  linewidth=2,

    # ax1.set_xlabel('time', labelpad=20); # fontsize = label_FontSize,
    ax1.set_ylabel(two_subplots_obj.y_label) # , labelpad=20); # fontsize = label_FontSize,

    ax1.tick_params(axis='y', which='major') #, labelsize = tick_FontSize)
    ax1.tick_params(axis='x', which='major') #, labelsize = tick_FontSize) # 
    ax1.legend(loc="upper right"); # , fontsize=12
    ax1.grid(True);


    #######
    #######  subplot 2
    #######
    ax2.plot(two_subplots_obj.x2, two_subplots_obj.y2, '-', label = two_subplots_obj.leg2, c = "dodgerblue")

    # ax2.set_xlabel('t') # , labelpad = 20); # fontsize = label_FontSize,
    ax2.set_ylabel(two_subplots_obj.y_label) # , labelpad = 20); # fontsize = label_FontSize,

    ax2.tick_params(axis = 'y', which = 'major') # , labelsize = tick_FontSize) #
    ax2.tick_params(axis = 'x', which = 'major') #, labelsize = tick_FontSize) # 

    ax2.legend(loc="upper right");
    ax2.grid(True);

    plt.xlim(two_subplots_obj.x_limits)



