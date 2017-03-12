#include <opencv2/opencv.hpp>  
#include <iostream>
#include <time.h>
#include "fftw3.h"

using namespace cv;
using namespace std;
#define Iteration_time 5
float const BOUND[] = {0.82, 0.4610, 0.2590,0.1450,0.0820 };
float const LEFT_LINE_P0[] = { 1.08715491833576,1.04404277230246,1.02198575258148,1.01061794069958,1.00489456979082};
float const LEFT_LINE_P1[] = {0.161872978061302,0.0670518176595489,0.0284143580342271,0.0120693414510977,0.00505846638184741 };
float const RIGHT_LINE_P0[] = {1.08715491833576,1.04404277230246,1.02198575258148,1.01061794069958,1.00489456979082 };
float const RIGHT_LINE_P1[] = {-0.161872978061305,-0.0670518176595504,-0.0284143580342275,-0.0120693414510978,-0.00505846638184776};

void circular_conv(Mat & const src, Mat & dst, Mat  & const kernel, Point anchor)
{
	Mat tmp,out_tmp;
	int kernel_col = kernel.cols;
	int kernel_row = kernel.rows;
	copyMakeBorder(src,tmp,anchor.y,kernel_row - 1 - anchor.y, anchor.x,kernel_col - 1 - anchor.x,BORDER_WRAP);
	filter2D(tmp,out_tmp,tmp.depth(),kernel,anchor);
	dst = Mat(out_tmp,Rect(anchor.x,anchor.y,src.cols,src.rows));
}

void divispectrum(Mat & const X1, Mat & const X2, Mat & X_out)
{
	//这里假定X2是实矩阵
	Mat X_split[2];
	split(X1,X_split);
	divide(X_split[0],X2,X_split[0]);
	divide(X_split[1],X2,X_split[1]);

	merge(X_split,2,X_out);
}

void Solve_w(Mat & const v, Mat & w, int beta_chose)
{
	float bound = BOUND[beta_chose];
	float left_line_p0 = LEFT_LINE_P0[beta_chose];
	float left_line_p1 = LEFT_LINE_P1[beta_chose];
	float right_line_p0 = RIGHT_LINE_P0[beta_chose];
	float right_line_p1 = RIGHT_LINE_P1[beta_chose];
	float tmp;

	int mat_col = v.cols;
	int mat_row = v.rows;
	w.create(v.size(),v.type());
	if(v.isContinuous() && w.isContinuous())
	{
		mat_col = mat_col*mat_row;
		mat_row = 1;
		
	}

	for(int i = 0;i<mat_row;i++)
	{
		const float* vptr = v.ptr<float>(i);
		float * wptr = w.ptr<float>(i);
		for(int j = 0; j<mat_col;j++)
		{
			tmp = vptr[j];
			if(tmp<-bound)
			{
				wptr[j] = tmp * left_line_p0 + left_line_p1;
			}
			else if(tmp<bound)
			{
				wptr[j] = 0;
			}
			else
			{
				wptr[j] = tmp * right_line_p0 + right_line_p1;
			}
		}
	}
}

void dft_fftw(Mat & const src, Mat & dst)
{
	int mat_row = src.rows;
	int mat_col = src.cols;

	float * data_tmp = (float *)fftwf_malloc(sizeof(float)*mat_row*mat_col);
	fftwf_complex * out_tmp = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex)*mat_row*(mat_col/2+1));
	fftwf_plan p = fftwf_plan_dft_r2c_2d(mat_row,mat_col,data_tmp,out_tmp, FFTW_ESTIMATE);

	int row = mat_row;
	int col = mat_col;
	
	if(src.isContinuous())
	{
		col = mat_col * mat_row;
		row = 1;
	}
	for(int i = 0; i<row;i++)
	{
		float* data_ptr = src.ptr<float>(i);
		memcpy((void *)(data_tmp + col*i),(void*)data_ptr ,sizeof(float)*col);
	}

	fftwf_execute(p);

	Mat out_mat[2];
	out_mat[0].create(mat_row,mat_col/2+1,CV_32F);out_mat[1].create(mat_row,mat_col/2+1,CV_32F);

	col = out_mat[0].cols; row = out_mat[0].rows;
	if(out_mat[0].isContinuous())
	{
		col = row*col;
		row = 1;
	}
	for(int i = 0; i<row ;i++)
	{
		float* data_ptr = out_mat[0].ptr<float>(i);
		for(int j = 0;j<col;j++)
		{
			data_ptr[j] = out_tmp[i*col+j][0];
		}
	}

	col = out_mat[1].cols; row = out_mat[1].rows;
	if(out_mat[1].isContinuous())
	{
		col = row*col;
		row = 1;
	}
	for(int i = 0; i<row ;i++)
	{
		float* data_ptr = out_mat[1].ptr<float>(i);
		for(int j = 0;j<col;j++)
		{
			data_ptr[j] = out_tmp[i*col+j][1];
		}
	}
	merge(out_mat,2,dst);

	fftwf_free(data_tmp);
	fftwf_free(out_tmp);
	fftwf_destroy_plan(p);
}

void idft_fftw(Mat& const src, Mat & dst)
{
	int dst_col = src.cols*2 - 1;
	int dst_row = src.rows;
	Mat src_tmp[2];

	fftwf_complex * data_tmp = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex)*src.rows*src.cols);
	float * out_tmp = (float *)fftwf_malloc(sizeof(float)*dst_col*dst_row);
	fftwf_plan p = fftwf_plan_dft_c2r_2d(dst_row,dst_col,data_tmp,out_tmp,FFTW_ESTIMATE);
	int col = src.cols;
	int row = src.rows;

	split(src,src_tmp);
	if(src_tmp[0].isContinuous())
	{
		col = col*row;
		row = 1;
	}
	for(int i = 0; i<row; i++)               //把Mat里的数据装入fftw指定的内存
	{
		float *real_ptr = src_tmp[0].ptr<float>(i);
		for(int j = 0; j<col; j++)
		{
			data_tmp[i*col+j][0] = real_ptr[j];
		}
	}
	if(src_tmp[1].isContinuous())
	{
		col = col*row;
		row = 1;
	}
	for(int i = 0; i<row; i++)               //把Mat里的数据装入fftw指定的内存
	{
		float *im_ptr = src_tmp[1].ptr<float>(i);
		for(int j = 0; j<col; j++)
		{
			data_tmp[i*col+j][1] = im_ptr[j];
		}
	}

	fftwf_execute(p);

	dst.create(dst_row,dst_col,CV_32F);
	row = dst_row;
	col = dst_col;

	if(dst.isContinuous())
	{
		col = row*col;
		row = 1;
	}
	for(int i = 0;i<row;i++)  //只对实部感兴趣
	{
		float * ptr = dst.ptr<float>(i);
		for(int j =0 ; j<col; j++)
		{
			ptr[j] = out_tmp[i*col+j]/dst_col/dst_row;
		}
	}

	fftwf_free(data_tmp);
	fftwf_free(out_tmp);
	fftwf_destroy_plan(p);
}

void fast_deblurring(Mat & const src_im, Mat & const kernel,Mat & yout)
{
	Mat tmp;
	float beta[Iteration_time] = { 2.828427,8.000000,22.627417,64.000000,181.019336 };  //beta是求解方程的稀疏变量，每次迭代增大
	float lambda = 5e2;
	float lambda_step = 2.828427124746190 ;//2*sqrt(2)

	Mat dx = (Mat_<float>(1,2) << 1,-1);
	Mat dy = (Mat_<float>(2,1) << 1,-1);
	Mat dx_flip = (Mat_<float>(1,2) << -1,1);
	Mat dy_flip = (Mat_<float>(2,1) << -1,1);
	
	Mat Denorm1,Denorm2,ky,dx_extended,dy_extended,k_extended;

	//计算求解公式中一些常数项
	//    ky  -- F(K)'*F(y)
	//    Denorm2  -- |F(K)|.^2
	//    Denorm1  -- |F(D^1)|.^2 + |F(D^2)|.^2

	copyMakeBorder(dx,dx_extended,0,src_im.rows - dx.rows,0, src_im.cols - dx.cols,BORDER_CONSTANT,0);
	copyMakeBorder(dy,dy_extended,0,src_im.rows - dy.rows,0, src_im.cols - dy.cols,BORDER_CONSTANT,0);
	copyMakeBorder(kernel,k_extended,0,src_im.rows - kernel.rows,0, src_im.cols - kernel.cols, BORDER_CONSTANT,0);

	circular_conv(src_im,tmp,kernel,Point(kernel.cols/2,kernel.rows/2));
	circular_conv(dx_extended,dx_extended,dx,Point(0,0)); //这里dx_extended和dy_extended做中间变量
	circular_conv(dy_extended,dy_extended,dy,Point(0,0));
	//dft(tmp,ky,DFT_COMPLEX_OUTPUT);
	dft_fftw(tmp,ky);

	Denorm1 = dx_extended + dy_extended;   //此处Denorm1 = dx * dx + dy * dy (*是correlate运算)
	//dft(Denorm1,Denorm1,DFT_COMPLEX_OUTPUT);
	//dft(k_extended,Denorm2,DFT_COMPLEX_OUTPUT);
	dft_fftw(Denorm1,Denorm1);
	dft_fftw(k_extended,Denorm2);

	Mat split_tmp[2];
	split(Denorm1,split_tmp);
	split_tmp[0].copyTo(Denorm1);
	split(Denorm2,split_tmp);
	magnitude(split_tmp[0],split_tmp[1],Denorm2);
	pow(Denorm2,2,Denorm2);

	yout = src_im; //yout 是迭代中的解，初始化为读入的图像,W是迭代过程中的一个中间量
	Mat youtx, youty;
	Mat Wx,Wy,Wxx,Wyy; //Wx Wy分别是W的水平，垂直差分，Wxx，Wyy则是对应的二阶差分
	Mat Denom;
	for(int i = 3; i<Iteration_time; i++)
	{
		Denom = beta[i]/lambda*(Denorm1) +Denorm2;  //K_2 此处为 |F(D^1)|.^2 + |F(D^2)|.^2+  beta/lambda * |F(K)|.^2	

		circular_conv(yout,youtx,dx_flip,Point(1,0)); //水平差分
		circular_conv(yout,youty,dy_flip,Point(0,1)); //竖直差分
	
		Solve_w(youtx,Wx,i);
		Solve_w(youty,Wy,i);

		circular_conv(Wx,Wxx,dx,Point(0,0));
		circular_conv(Wy,Wyy,dy,Point(0,0));

		Wxx += Wyy;
		
		//dft(Wxx,Wxx,DFT_COMPLEX_OUTPUT); //此处Wxx 是W水平，垂直二阶差分和的傅里叶变换		
		dft_fftw(Wxx,Wxx);

		yout = ( ky + beta[i]/lambda* Wxx);
		//cout<<yout<<endl<<endl;

		Mat tmp2;
		divispectrum(yout,Denom,tmp2);

		//dft(tmp2,yout,DFT_INVERSE|DFT_SCALE);
		idft_fftw(tmp2,yout);

		lambda *= lambda_step;
	}
}
int main()
{
	Mat src_im,kernel,out[3],src[3],imout;
	
	imread("TM.BMP",CV_LOAD_IMAGE_COLOR    ).convertTo(src_im,CV_32F);
	src_im /=255;
	//Mat(src_im,Rect(0,0,10,10)).copyTo(src_im);
	imread("kernel4.png",CV_LOAD_IMAGE_GRAYSCALE ).convertTo(kernel,CV_32F);
	//Mat(kernel,Rect(0,0,5,5)).copyTo(kernel);
	kernel /= 255;

	clock_t start,end;
	
	start = clock();
		
		split(src_im,src);
		cout<<src_im.channels()<<endl;
		for(int i =0;i<src_im.channels();i++)
		{
			fast_deblurring(src[i],kernel,out[i]);
		}
		merge(out,src_im.channels(),imout);

	end = clock();
	cout<<difftime(end,start)/CLOCKS_PER_SEC<<endl;

	imshow("23333",imout);
	imout *= 255;
	imout.convertTo(imout,CV_8U);
	imwrite("out.bmp",imout);
	waitKey(0);
}
