A DVD rental company needs your help! They want to figure out how many days a customer will rent a DVD for based on some features and has approached you for help. They want you to try out some regression models which will help predict the number of days a customer will rent a DVD for. The company wants a model which yeilds a MSE of 3 or less on a test set. The model you make will help the company become more efficient inventory planning.

The data they provided is in the csv file rental_info.csv. It has the following features:

"rental_date": The date (and time) the customer rents the DVD.
"return_date": The date (and time) the customer returns the DVD.
"amount": The amount paid by the customer for renting the DVD.
"amount_2": The square of "amount".
"rental_rate": The rate at which the DVD is rented for.
"rental_rate_2": The square of "rental_rate".
"release_year": The year the movie being rented was released.
"length": Lenght of the movie being rented, in minuites.
"length_2": The square of "length".
"replacement_cost": The amount it will cost the company to replace the DVD.
"special_features": Any special features, for example trailers/deleted scenes that the DVD also has.
"NC-17", "PG", "PG-13", "R": These columns are dummy variables of the rating of the movie. It takes the value 1 if the move is rated as the column name and 0 otherwise. For your convinience, the reference dummy has already been dropped.
