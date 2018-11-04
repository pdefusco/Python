#calling the format method on a predefined string composed by slots for printing variables:

#use formatter to hold the four variables
formatter = "{} {} {} {}"

print(formatter.format(1,2,3,4))
print(formatter.format('text1', 'moretext', 'text3', 'text4'))
print(formatter.format(formatter.format(1,2,3,4), formatter.format(2,3,4,5), formatter.format('var1','2','3','var4'), formatter.format('ein', 'zwei', 'drei', 'vier')))
