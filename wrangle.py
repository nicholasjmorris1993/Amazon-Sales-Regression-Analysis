def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import time
import numpy as np
import pandas as pd


def prepare(df):
    data = Prepare(df)
    print("Data Wrangling:")
    start = time.time()
    data.drop()
    data.product_name()
    data.category()
    data.actual_price()
    data.discount_percentage()
    data.rating()
    data.rating_count()
    data.aggregate()
    data.shuffle()
    end = time.time()
    data.run_time(start, end)

    return data.df
    
    
class Prepare:
    def __init__(self, df):
        self.df = df  # dataset
    
    def drop(self):
        print("> Removing Unnecessary Columns")
        self.df = self.df.drop(columns=[
            "discounted_price", 
            "user_id", 
            "user_name", 
            "review_id", 
            "img_link", 
            "product_link",
        ])
    
    def product_name(self):
        print("> Transforming Product Name")
        # get the first word from product_name
        word = list()
        for item in self.df["product_name"]:
            word.append(item.split(" ")[0])
        self.df["product_name"] = word

    def category(self):
        print("> Transforming Category")
        # get the first section from category
        section = list()
        for item in self.df["category"]:
            section.append(item.split("|")[0])
        self.df["category"] = section

    def actual_price(self):
        print("> Transforming Actual Price")
        # convert actual_price to a float
        self.df["actual_price"] = self.df["actual_price"].str.replace("₹", "")
        self.df["actual_price"] = self.df["actual_price"].str.replace(",", "")
        self.df["actual_price"] = self.df["actual_price"].astype(float)

    def discount_percentage(self):
        print("> Transforming Discount Percentage")
        # convert discount_percentage to a float
        self.df["discount_percentage"] = self.df["discount_percentage"].str.replace("%", "")
        self.df["discount_percentage"] = self.df["discount_percentage"].astype(float) / 100
        
    def rating(self):
        print("> Transforming Rating")
        # convert rating to a float
        self.df["rating"] = self.df["rating"].str.replace("|", "NaN")
        self.df["rating"] = self.df["rating"].astype(float)

    def rating_count(self):
        print("> Transforming Rating Count")
        # convert rating_count to a float
        self.df["rating_count"] = self.df["rating_count"].str.replace(",", "")
        self.df["rating_count"] = self.df["rating_count"].astype(float)
    
    def aggregate(self):
        print("> Aggregating By Product ID")
        # aggregate the strings by product_id
        strings = self.df.groupby("product_id")[[
            "product_id", 
            "product_name", 
            "category", 
            "about_product", 
            "review_title", 
            "review_content",
        ]].head(1)
        strings = strings.sort_values(by="product_id").reset_index(drop=True)

        # aggregate the numbers by product_id
        self.df = self.df.groupby("product_id").agg({
            "actual_price": "mean",
            "discount_percentage": "mean",
            "rating": "mean",
            "rating_count": "mean",
        }).reset_index()
        self.df = self.df.sort_values(by="product_id").reset_index(drop=True)

        # add the strings back to the data
        self.df = pd.concat([strings, self.df.drop(columns="product_id")], axis="columns")

        # remove the product_id
        self.df = self.df.drop(columns="product_id")
    
    def shuffle(self):
        print("> Shuffling The Data")
        self.df = self.df.sample(frac=1, random_state=0).reset_index(drop=True)

    def run_time(self, start, end):
        duration = end - start
        if duration < 60:
            duration = f"{round(duration, 2)} Seconds"
        elif duration < 3600:
            duration = f"{round(duration / 60, 2)} Minutes"
        else:
            duration = f"{round(duration / 3600, 2)} Hours"
        print(duration)
