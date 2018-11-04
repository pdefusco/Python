#Store Inventory Class

class Inventory(object):
    def __init__(self, company, department):
        self.company = company
        self.department = department

class InfoSys(object):
    def __init__(self, vendor, cost):
        self.vendor = vendor
        self.cost = cost

class ClothingInventory(Inventory, InfoSys):

    def __init__(self, items = [], inventory_limit = 0):
        self.items = items
        self.inventory_limit = inventory_limit
        self.InfoSys = InfoSys("InfoCo", "High")

    def set_attribute(self):
        inp = input('> ')
        self.items.append(inp)

    def get_attribute(self):
        return self.items

    def set_inventory_limit(self):
        print("Please enter number of clothing items you would like to add: ")
        inp = int(input('> '))
        self.inventory_limit = inp

    def get_inventory_limit(self):
        print(f"The inventory limit is currently set to: {self.inventory_limit}")

    def print_infosys(self):
        print("Information System attributes:")
        print(f"Vendor: {self.InfoSys.vendor}")
        print(f"Cost: {self.InfoSys.cost}")

    def change_infosys_attributes(self):
        print("Currently the Information System used is: ")
        print(f"Vendor:  {self.InfoSys.vendor}")
        print(f"Cost: {self.InfoSys.cost}")
        print("Would you like to change the values? Y or N")
        inp = input('> ')

        if inp == 'Y':
            print("Please enter the new values")
            print("Vendor: ")
            input_vendor = input('>')
            print("Cost: ")
            input_cost = input('>')
        elif inp == 'N':
            print("You have opted to stay with the current values")
            exit()
        else:
            print("Please enter either Y or N")
            change_infosys_attributes(self)

        return input_vendor, input_cost

    def setnew_infosys_attributes(self, input_vendor, input_cost):
        self.InfoSys.input_vendor = input_vendor
        self.InfoSys.input_cost = input_cost
        print(f"The new values for vendor and cost have been set to {input_vendor} and {input_cost} respectively")
