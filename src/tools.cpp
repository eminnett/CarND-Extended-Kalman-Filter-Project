#include <iostream>
#include <cmath>
#include "tools.h"

using namespace std;
using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
    VectorXd rmse(4);
    rmse << 0,0,0,0;

    // check the validity of the following inputs:
    //  * the estimation vector size should not be zero
    //  * the estimation vector size should equal ground truth vector size
    if (estimations.size() != ground_truth.size() || estimations.size() == 0) {
        cout << "Invalid estimation or ground_truth data" << endl;
        return rmse;
    }

    //accumulate squared residuals
    for (unsigned int i=0; i < estimations.size(); ++i) {

        VectorXd residual = estimations[i] - ground_truth[i];

        //coefficient-wise multiplication
        residual = residual.array()*residual.array();
        rmse += residual;
    }

    //calculate the mean
    rmse = rmse/estimations.size();

    //calculate the squared root
    rmse = rmse.array().sqrt();

    return rmse;
}
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
    MatrixXd Hj(3,4);
    Hj << 0, 0, 0, 0,
          0, 0, 0, 0,
          0, 0, 0, 0;

    //recover state parameters
    float px = x_state(0);
    float py = x_state(1);
    float vx = x_state(2);
    float vy = x_state(3);

    //check division by zero
    float px2_plus_py2 = px*px+py*py;
    if (px2_plus_py2 == 0) {
        cout << "Division by 0 avoided." << endl;
        px2_plus_py2 = 0.00000001;
    }

    float position_vec_magnitude = sqrt(px2_plus_py2);
    float position_vec_magnitude_cubed = pow(position_vec_magnitude, 3);

    //compute the Jacobian matrix
    float drho_dpx = px / position_vec_magnitude;
    float drho_dpy = py / position_vec_magnitude;

    float dphi_dpx = - py / px2_plus_py2;
    float dphi_dpy = px / px2_plus_py2;

    float drhodot_dpx = py*(vx*py-vy*px)/position_vec_magnitude_cubed;
    float drhodot_dpy = px*(vy*px-vx*py)/position_vec_magnitude_cubed;
    float drhodot_dvx = drho_dpx;
    float drhodot_dvy = drho_dpy;

    Hj << drho_dpx,    drho_dpy,    0,    0,
          dphi_dpx,    dphi_dpy,    0,    0,
          drhodot_dpx, drhodot_dpy, drhodot_dvx, drhodot_dvy;

    return Hj;
}
