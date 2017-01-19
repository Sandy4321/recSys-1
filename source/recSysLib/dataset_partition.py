class DataPartition():
    "Class that compute the dataset splitting for the performance evaluation"

    def __init__(self, data):
        self.data = data
        self.data_rows = data.shape[0]
        self.data_cols = data.shape[1]

    def toString(self):
        print("DataPartition Class. Dataset rows: {}  cols: {}".format(self.data.shape[0],self.data.shape[1]))

    def split_cross(self, train_perc_row=0.9, train_perc_col=0.9,verbose=0):
        self.train_items_row = int(train_perc_row * self.data_rows)
        self.train_items_col = int(train_perc_col * self.data_cols)
        if verbose > 0:
            print("Splitting in train and test!")
            print("Number of items by row:{} , by col: {}".format(self.train_items_row,self.train_items_col))

    def get_upLeft_matrix(self):
        return self.data[0:self.train_items_row,0:self.train_items_col]

    def get_lowRight_matrix(self):
        return self.data[self.train_items_row:self.data_rows,self.train_items_col:self.data_cols]

    def get_upRight_matrix(self):
        return self.data[0:self.train_items_row,self.train_items_col:self.data_cols]

    def get_lowLeft_matrix(self):
        return self.data[self.train_items_row:self.data_rows,0:self.train_items_col]
