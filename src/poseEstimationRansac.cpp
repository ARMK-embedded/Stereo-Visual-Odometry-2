//x = [alpha, beta, gamma, x, y, z];

/*given the 3d point correspondences in two consecutive frames 
i.e previous frame to current frame, use Gauss Newton Minimization 
to find maximum likelihood estimate of the camera 
relative transformation , 6DOF pose*/

//z: is measured 3d point location
//h: measurement map as a function of state (camera pose)


//compute the measurement jacobians first
#include "SVO.h"
#include <random>
#include <utility>
#include <math.h>
#define _USE_MATH_DEFINES
//# define M_PI           3.14159265358979323846  /* pi */

using std::mt19937;
using std::uniform_int_distribution;

using namespace std;


bool traverse(vector<int> &result, int r, int range)
{
	bool found =false;
	for(int i =0; i< range; ++i )
	{
		if(r == result[i])
		{
			found = true;
		}


	}
	return found;
}

typedef uniform_int_distribution<> uniform;
mt19937 gen(0);


void randGenerator(vector<int> &result, int range)
{
	uniform point_gen(0, range);
	for(int i = 0; i < int(result.size()); ++i)
	{
	   int r;

   	 	do 
	   {
	   	r = point_gen(gen);
	   }while(traverse(result,r, i));
		       
	   result[i] = r; 
	   //cout << "random_no: " << i<< " : " <<  result[i] <<  endl;
	  
	}

}




struct  RansacId
{
	vector< vector<int> > RansacPointIds;
	//vector<double> RansacMedians;
	vector<pair<double,int> >RansacMedians;
	
};


void eulerR(double &alpha, double &beta, double &gamma, Matrix3d &Rz,
				Matrix3d &Ry, Matrix3d &Rx)
{
	alpha = fmod(alpha , 2.0*M_PI);
	beta = fmod(beta, 2.0*M_PI);
	gamma = fmod(gamma , 2.0*M_PI);
	Rz << cos(alpha), -sin(alpha), 0,
			sin(alpha), cos(alpha),0,
			0, 		0,		1;

	Ry << cos(beta), 0, sin(beta),
		   0,      1,    0,
		-sin(beta),0, cos(beta);

	Rx << 1, 0, 0,
		  0, cos(gamma), -sin(gamma),
		   0, sin(gamma), cos(gamma);
}



void SVO::GaussNewton(vector<Vector3d> &p1, vector<Vector3d> &p2, VectorXd &XGN, MatrixXd &PGN)
{
	//cout << "in GaussNewton" << endl;
	
	//initial estimate
	XGN = this->X;       // VectorXd::Zero(6,1);
	PGN = this->P;//MatrixXd::Zero(6,6);

	for (int iter =0; iter< 10; ++iter)
	{
		//cout << "in iter " << endl;
		MatrixXd HQH = MatrixXd::Zero(6,6);
		MatrixXd HQz = MatrixXd::Zero(6,1);
		//cout << "p1 size" << p1.size() << endl;
		for (int i=0; i< int(p1.size()); ++i)
		{
			//cout << "in i" << endl;
			//double alpha = XGN(0);
			//double beta = XGN(1);
			//double gamma = XGN(2); 
			Vector3d pos = XGN.segment<3>(3);
			//cout << "after XGN" << endl;
			Matrix3d Rz, Ry, Rx;
			eulerR(XGN(0) /*alpha*/,XGN(1) /*beta*/,XGN(2)/*gamma*/, Rz, Ry, Rx);
			//cout << "after eulerR" << endl;
			Matrix3d J1;
			J1 << 0, -1, 0,
				  1, 0, 0,
				  0, 0, 0;

			Matrix3d J2;
			J2  << 0, 0, 1,
						   0, 0, 0,
						   -1, 0, 0;

			Matrix3d J3 ;
			J3 << 0, 0, 0,
						   0, 0, -1,
						   0, 1, 0;
			//cout << "before halpha" << endl;

			MatrixXd halpha = Rz*J1*Ry*Rx*p2[i];
			MatrixXd hbeta = Rz*Ry*J2*Rx*p2[i];
			MatrixXd hgamma = Rz*Ry*Rx*J3*p2[i];
			MatrixXd Hp = MatrixXd::Identity(3,3);

			//cout << " after Hp" << endl;

			//MatrixXd H;
			H << halpha, hbeta, hgamma, Hp;

			Vector3d z = p1[i];
			Vector3d hx = Rz*Ry*Rx*p2[i]+ pos;
			//cout << "after hx" << endl;
			HQH = HQH + H.transpose()*Q.inverse()*H;
			//cout << "after HQH" << endl;
			//cout << "Z: " << z.transpose() << " , " << "hx: " << hx.transpose() << endl;
			//cout << H.transpose() << endl;
			//cout << Q.inverse() << endl;
			//cout << H.transpose()*Q.inverse()*(z- hx) << endl;
			HQz = HQz + H.transpose()*Q.inverse()*(z- hx);

			//cout << "end of i" << endl;


		}// end of i loop
		 //P and X are automatically stored as they are SVO members
		PGN = HQH.inverse(); 
		//cout << "P*HQz: " << P*HQz << endl;
		XGN = XGN + PGN*HQz;


	}// end of GN iteration loop


}



void SVO::RansacGN(vector<Vector3d> &p1, vector<Vector3d> &p2)
{
	//cout << "entered RansacGN function" << endl;
	VectorXd XGN;
	MatrixXd PGN;
	Matrix3d Rz, Ry, Rx;
	struct RansacId RID;
	vector<Vector3d> p1s, p2s;
	int range = p1.size();
	int min_points = 75;
	
	cout << "jacobian H: " << H << endl;

	for (int i=0; i< 500; ++i)
	{
		//choose 5 points at random
		//cout << "in main ransac loop " << endl;
		vector<int> pointIds(min_points,0);
		randGenerator(pointIds, range);

		p1s.clear(); p2s.clear();

		for (int j=0; j< int(pointIds.size()); ++j)
		{
			p1s.push_back(p1[pointIds[j]]);
			p2s.push_back(p2[pointIds[j]]);
	
		}


	    GaussNewton(p1s, p2s, XGN, PGN);
	    //cout << "GN function iteration: " << i << endl;
		vector<double> squaredResiduals(min_points,0);

		//calculate residuals
		for(int j=0; j < min_points; ++j)
		{
			//cout << "in residual loop" << endl;
			eulerR(XGN(0), XGN(1), XGN(2), Rz, Ry, Rx);

			Vector3d residual = p1[pointIds[j]] - Rz*Ry*Rx*p2[j]+ XGN.segment<3>(3);
			//cout << "after residual " << endl;
			Matrix3d S = H*PGN*H.transpose() + Q;
			//cout << "after S" << endl;
			double squaredResidual = residual.transpose()*S.inverse()*residual ;
			//cout << "after squaredResidual " << endl;
			squaredResiduals[j] = squaredResidual;
		}
		//cout << "after residual loop" << endl;

		//compute the median of the squaredResiduals
		//std::nth_element(v.begin(), v.begin() + v.size()/2, v.end());
    	
		//RID.medians[i] = median(squaredResiduals);
		std::nth_element(squaredResiduals.begin(), squaredResiduals.begin() + squaredResiduals.size()/2, squaredResiduals.end());
		RID.RansacMedians.push_back(make_pair(squaredResiduals[squaredResiduals.size()/2],i)) ;
		//std::cout << "The median is " << RID.RansacMedians[i].first << '\n';
		RID.RansacPointIds.push_back(pointIds);
	
	}// end of ransac for loop
	//cout << "after ransac loop " << endl;
	std::sort(RID.RansacMedians.begin(), RID.RansacMedians.end()); //keep track of index
	int index = RID.RansacMedians[0].second;
	//save the index of point set with least median
	p1s.clear(); p2s.clear();
	for (int j=0; j<int(RID.RansacPointIds[index].size()); ++j)
	{
		p1s.push_back(p1[RID.RansacPointIds[index][j]]);
		p2s.push_back(p2[RID.RansacPointIds[index][j]]);

	}


	GaussNewton(p1s, p2s, XGN, PGN);

	if ((XGN.segment<3>(3).norm() < 100) && ((X.segment<3>(3) - XGN.segment<3>(3)).norm() < 15))
	{
		this->X = XGN;
		//cout << "X : " << X.transpose() << endl;
	 	this->P = PGN;
	 	eulerR(XGN(0), XGN(1), XGN(2), Rz, Ry, Rx);
	 	Matrix3d R = Rz*Ry*Rx;
	 	cout << "determinant of R: " << R.determinant() << endl;
	 	MatrixXd POSE_REL = MatrixXd::Zero(4,4);
		POSE_REL.block<3,3>(0,0) =  R;
		POSE_REL.block<3,1>(0,3) = XGN.segment<3>(3);
		POSE_REL(3,3) = 1.0;
		cout << "POSE_REL: " << POSE_REL <<  endl;

		POSE = POSE_REL*POSE;
	
	}



	



}