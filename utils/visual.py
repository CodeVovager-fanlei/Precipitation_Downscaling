import matplotlib.pyplot as plt
import pathlib


def setlabel(ax, label, prop, bbox_to_anchor):
    ax.text(
        bbox_to_anchor[0], bbox_to_anchor[1],
        label,
        transform=ax.transAxes,
        fontsize=prop['size'],
        fontweight=prop['weight'],
        # fontfamily=fontfamily,
        va='top'
    )


def showfig(fig, close=False):
    '''Show the figure

    Parameters
    ----------
    fig : matplotlib.pyplot.figure
        The matplotlib figure object

    close : bool
        if True, close the figure automatically

    See Also
    --------
    pyleoclim.utils.plotting.savefig : saves a figure to a specific path
    pyleoclim.utils.plotting.in_notebook: Functions to sense a notebook environment

    '''
    # if in_notebook():
    #     try:
    #         from IPython.display import display
    #     except ImportError as error:
    #         # Output expected ImportErrors.
    #         print(f'{error.__class__.__name__}: {error.message}')
    #
    #     display(fig)
    #
    # else:
    #     plt.show()

    plt.show()

    if close:
        closefig(fig)


def closefig(fig=None):
    '''Show the figure

    Parameters
    ----------
    fig : matplotlib.pyplot.figure
        The matplotlib figure object

    See Also
    --------
    pyleoclim.utils.plotting.savefig : saves a figure to a specific path
    pyleoclim.utils.plotting.in_notebook: Functions to sense a notebook environment

    '''
    if fig is not None:
        plt.close(fig)
    else:
        plt.close()


def savefig(fig, path=None, settings={}, verbose=True):
    ''' Save a figure to a path

    Parameters
    ----------
    fig : matplotlib.pyplot.figure
        the figure to save
    path : str
        the path to save the figure, can be ignored and specify in "settings" instead
    settings : dict
        the dictionary of arguments for plt.savefig(); some notes below:
        - "path" must be specified in settings if not assigned with the keyword argument;
          it can be any existed or non-existed path, with or without a suffix;
          if the suffix is not given in "path", it will follow "format"
        - "format" can be one of {"pdf", "eps", "png", "ps"}

    See Also
    --------

    pyleoclim.utils.plotting.showfig : returns a visual of the figure.
    '''
    if path is None and 'path' not in settings:
        raise ValueError('"path" must be specified, either with the keyword argument or be specified in `settings`!')

    savefig_args = {'bbox_inches': 'tight', 'path': path}
    savefig_args.update(settings)

    path = pathlib.Path(savefig_args['path'])
    savefig_args.pop('path')

    dirpath = path.parent
    if not dirpath.exists():
        dirpath.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f'Directory created at: "{dirpath}"')

    path_str = str(path)
    if path.suffix not in ['.eps', '.pdf', '.png', '.ps']:
        path = pathlib.Path(f'{path_str}.pdf')

    fig.savefig(path_str, **savefig_args)
    plt.close(fig)

    if verbose:
        print(f'Figure saved at: "{str(path)}"')
