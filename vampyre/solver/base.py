class Solver(object):
    """
    Base class for solver
    
    :param hist_list:  String list of variables to save the history of.  
       These will be saved in the :code:`hist_dict` dictionary
    """    
    def __init__(self, hist_list):
        self.hist_list = hist_list
        self.init_hist()
        
    def init_hist(self):
        """
        Initializes the history dictionary.
        
        This should be called any time :code:`hist_list` has been changed
        """        
        # Initialize the dictionary with an empty list for each item
        self.hist_dict = {}
        for attr in self.hist_list:
            self.hist_dict[attr] = []
            
            
    def save_hist(self):
        """
        Save the items in the history dictionary
        """
        for attr in self.hist_list:
            self.hist_dict[attr].append(getattr(self,attr))
            
        
