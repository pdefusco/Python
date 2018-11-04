##Main application module
##Used to run application

##This application will allow to manage and store clothing inventory

from Inventory import Inventory, ClothingInventory

store1 = ClothingInventory()

store1.set_inventory_limit()

count = 0
while count < store1.inventory_limit:
    print(store1.get_attribute())
    store1.set_attribute()
    count += 1

print(store1.get_attribute())

store1.print_infosys()
input_cost, input_vendor = store1.change_infosys_attributes()
store1.setnew_infosys_attributes(input_cost, input_vendor)
