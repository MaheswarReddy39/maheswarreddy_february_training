# Online Shopping Discount System
customer_name = input("Enter customer name: ")
price = float(input("Enter product price: "))
premium = input("Is the customer a premium member? (True/False): ")
coupon_code = input("Enter coupon code: ")
if premium.lower() == "true":
    premium = True 
else:
    False
discount = 0
if price > 5000 and premium:
    discount = 0.20 * price
elif premium or coupon_code == "SAVE10":
    discount = 0.10 * price
final_price = price - discount
print("Bill Details :")
print(f"Customer Name: {customer_name}")
print(f"Original Price: ₹{price:.2f}")
print(f"Discount Applied: ₹{discount:.2f}")
print(f"Final Price: ₹{final_price:.2f}")
if premium:
    print("Premium benefits applied")
