import csv
import pandas as pd 

# danh sach cac duoi url
checkList = [".gif", ".jpg", ".jpeg", ".GIF", ".JPG" , ".JPEG"] 
# du lieu dau vao
INPUT =  'output_http_csic_2010_weka_with_duplications_RAW-RFC2616_escd_v02_norm_test.csv'
#chuong trinh loai bo cac url anh
def remove_image_url(file):
	#load du lieu tu file
	raw_data = pd.read_csv(file, low_memory=False)
	for i in range(len(raw_data)):
		for checkData in checkList:
			#check url list
			if checkData in raw_data["url"][i]:
				#loai url
				raw_data = raw_data.drop(i, axis=0)
				break
	#ghi ra file moi
	raw_data.to_csv("Parsed_" + file,index = False , sep = ',' , encoding = 'utf-8')

def main():
	remove_image_url(INPUT)
	print("remove image url success")
	

if __name__ == '__main__':
    main()


