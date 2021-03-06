# ISD_yolo_dual

yolo-dual on ISD dataset for paper 

"Boost Infrared Moving Aircraft Detection Performance by Using Fast Homography Estimation and Dual Input Object Detection Network" (under Submission)

#author&copyright#

	Dawei Li
	Ph.D student of Beijing institute of Technolegy
	lidawei0124@gmail.com


#declaration#
	
	the code of  "dual input object detection network" part is modifed based on yolov5s model(https://github.com/ultralytics/yolov5)
	
#datasets#

	origin ISD dataset(22 sequences): http://csdata.org/en/p/387/
	
	dataset ISD in paper: 
	google https://drive.google.com/file/d/1DZAsvZHVIb4Ro-571weHHd8cmq4iuEAp/view?usp=sharing
	baidupan link：https://pan.baidu.com/s/1Yd7ZCWk5CU5fV3Ij55Xy-g code：1daf
	
	dataset ISDMINI  in paper: 
	google https://drive.google.com/file/d/1QTtywg44lkoe7hs3li8foZPZHmtKolTK/view?usp=sharing
	baidupan link：https://pan.baidu.com/s/1copt0SliP_KwqMQT3ssawA code：77fy 

#first step#

	download dataset from above urls, then unzip and put folders into path "img_registration"


#(option)train "the fast homography estimation network"#

*download coco dataset if you want to train "the fast homography estimation network" from scratch:

	cd img_registration
	python train_fast_homography.py

*or you can just use the pretrained weight file in path

	"img_registration/fast_homography_acc0.07fps168.pth"


#(option)train "dual input object detection network"#

*if you want to train "dual input object detection network" from scratch:

	python train.py

*or you can just use the pretrained weight files in path :

	"save/isd_yolo_dual" for yolo-dual on ISD
	"save/isd_yolo_single" for yolo-single on ISD
	"save/isd_yolo_stack" for yolo-stack on ISD
	"save/isdmini_yolo_dual" for yolo-dual on ISDMINI
	"save/isdmini_yolo_single" for yolo-single on ISDMINI
	"save/isdmini_yolo_stack" for yolo-stack on ISDMINI


#test "the fast homography estimation network"#

	cd img_registration
	python test_homography_compare.py


#test the infrared moving aircraft detection algorithm in paper#

	python demo.py

########Benchmark###########

ON ISD:


	methods			Precision Recall AP50 AP50:95
	Yolo5-Single		0.813	0.606	0.685	0.444
	Yolo5-Stack		0.861	0.829	0.86	0.579
	Yolo5-Dual(ours)	0.905	0.862	0.921	0.6


ON ISDMINI:


					Methods     data5/AP50 data13/AP50 data15/AP50 data8/AP50 data18/AP50 FPS						
	Quoted from paper of ISDet	YOLO V3		0.725	0.646	0.767	0.803	0.801	63
					RFBnet		0.763	0.622	0.784	0.837	0.842	71
					RefineDet	0.881	0.742	0.82	0.852	0.884	45
					ISTDet*		0.916	0.715	0.818	0.79	0.837	165
					ISTDet		0.938	0.794	0.883	0.896	0.928	76
	Tested in this paper		Yolo5-Single	0.995	0.876	0.862	0.995	0.73	108
					Yolo5-Stack	0.994	0.741	0.993	0.995	0.996	57
					Yolo5-Dual	0.995	0.99	0.992	0.995	0.995	55
