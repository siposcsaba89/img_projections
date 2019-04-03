#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/ccalib/omnidir.hpp>

struct Camera
{
    Camera() {}
    Camera(const std::string &  cali_file)
    {
        load_from_file(cali_file);
    }
    void load_from_file(const std::string &  cali_file)
    {
        cv::FileStorage fs2(cali_file, cv::FileStorage::READ);
        fs2["model_type"] >> model_type;
        fs2["camera_name"] >> camera_name;
        fs2["image_width"] >> img_width;
        fs2["image_height"] >> img_height;
        fs2["mirror_parameters"]["xi"] >> xi;
        dist_coeffs.resize(4);
        fs2["distortion_parameters"]["k1"] >> dist_coeffs[0];
        fs2["distortion_parameters"]["k2"] >> dist_coeffs[1];
        fs2["distortion_parameters"]["p1"] >> dist_coeffs[2];
        fs2["distortion_parameters"]["p2"] >> dist_coeffs[3];

        K = Eigen::Matrix3f::Identity();
        fs2["projection_parameters"]["gamma1"] >> K(0, 0);
        fs2["projection_parameters"]["gamma2"] >> K(1, 1);
        fs2["projection_parameters"]["u0"] >> K(0, 2);
        fs2["projection_parameters"]["v0"] >> K(1, 2);
    }

    void distortion(float mx_u, float my_u,
                                float *dx_u, float *dy_u) {
        float mx2_u, my2_u, mxy_u, rho2_u, rad_dist_u;

        mx2_u = mx_u * mx_u;
        my2_u = my_u * my_u;
        mxy_u = mx_u * my_u;
        rho2_u = mx2_u + my2_u;
        rad_dist_u = dist_coeffs[0] * rho2_u + dist_coeffs[1] * rho2_u*rho2_u;
        *dx_u = mx_u * rad_dist_u + 2 * dist_coeffs[2] *mxy_u + dist_coeffs[3] * (rho2_u + 2 * mx2_u);
        *dy_u = my_u * rad_dist_u + 2 * dist_coeffs[3] *mxy_u + dist_coeffs[2] * (rho2_u + 2 * my2_u);
    }

    Eigen::Vector3f projectPoint(const Eigen::Vector3f & p)
    {
        Eigen::Vector3f ret;
        // Project points to the normalised plane
        float norm = p.norm();
        float z = p.z() + xi * norm;

        
        float mx_u = p.x() / z;
        float my_u = p.y() / z;

        {
            // Apply distortion
            float dx_u, dy_u;
            distortion(mx_u, my_u, &dx_u, &dy_u);
            ret.x() = mx_u + dx_u;
            ret.y() = my_u + dy_u;
            ret.z() = 1.0f;
        }

        // Apply generalised projection matrix
        // Matlab points start at 1
        return K * ret;
    }

    int32_t img_width;
    int32_t img_height;
    std::string camera_name;
    std::string model_type;
    Eigen::Matrix3f K;
    float xi;
    std::vector<float> dist_coeffs; /// k1 k2 p1 p2
    Eigen::Matrix4f T_glob_cam = Eigen::Matrix4f::Identity();
};

cv::Vec2d ocamToImage(const cv::Vec3d & w_p, const std::vector<double> & inv_poly,
    cv::Point2d principal_point)
{
    cv::Vec3d tmp = w_p;
    tmp /= tmp[2];
    
    float dist = sqrt(tmp[0] * tmp[0] + tmp[1] * tmp[1]);

    float rho = atan2f(-1, dist);
    float tmp_rho = 1;
    float polyval = 0;
    for (size_t k = 0; k < inv_poly.size(); ++k)
    {
        float coeff = inv_poly[k];
        polyval += coeff * tmp_rho;
        tmp_rho *= rho;
    }
    cv::Vec2d ret;
    ret[0] = tmp[0] / dist * polyval;
    ret[1] = tmp[1] / dist * polyval;

    ret[0] += principal_point.x;
    ret[1] += principal_point.y;

    return ret;
}


/**
 * \brief Lifts a point from the image plane to its projective ray
 *
 * \param p image coordinates
 * \param P coordinates of the projective ray
 */
Eigen::Vector3d ocamToWordRay(const Eigen::Vector2d & pc, const std::vector<double> poly,
    const Eigen::Vector2d& p)
{
    // Relative to Center
    Eigen::Vector2d xc(p[0] - pc.x(), p[1] - pc.y());

    // Affine Transformation
    // xc_a = inv(A) * xc;
    double phi = std::sqrt(xc[0] * xc[0] + xc[1] * xc[1]);
    double phi_i = 1.0;
    double z = 0.0;

    for (int i = 0; i < (int)poly.size(); i++)
    {
        z += phi_i * poly[i];
        phi_i *= phi;
    }

    return Eigen::Vector3d( xc[0], xc[1], -z);
}



/**
 * \brief Project a 3D point (\a x,\a y,\a z) to the image plane in (\a u,\a v)
 *
 * \param P 3D point coordinates
 * \param p return value, contains the image point coordinates
 */

Eigen::Vector2d WorldToPlane(const Eigen::Vector3d& P, const std::vector<double> & inv_poly,
    const Eigen::Vector2d & pp)
{
    double norm = std::sqrt(P[0] * P[0] + P[1] * P[1]);
    double theta = std::atan2(-P[2], norm);
    double rho = 0.0;
    double theta_i = 1.0;

    for (int i = 0; i < (int)inv_poly.size(); i++)
    {
        rho += theta_i * inv_poly[i];
        theta_i *= theta;
    }

    double invNorm = 1.0 / norm;
    Eigen::Vector2d xn(
        P[0] * invNorm * rho,
        P[1] * invNorm * rho
    );

    return Eigen::Vector2d(
        xn[0] + pp.x(),
        xn[1] + pp.y());
}



int main(int argc, const char * argv[])
{
    std::string in_camera_file =
        "f:/tmp/autokali/jarvis/tmp/calib_orig/tmp/B_MIDRANGE_C.yaml";
    std::string out_file_name = 
        "f:/tmp/autokali/Pumba/input/calib/camera_3_calib.yaml";

    //cv::FileStorage fs(in_camera_file, cv::FileStorage::READ);
    //int dev_id;
    //dev_id <<fs["sensors"][0]["device_id"];
    bool mei = true;
    Eigen::Vector2d principal_point = { 658.058016768, 363.4490040319999 };
    std::vector<double> poly = { -622.9045934690578, 0, 0.0009136929368614688, -1.072594321111573e-06, 1.70188246008661e-09 };
    std::vector<double> poly_inv = { 759.2918034243791, 302.2036682636095, -38.28626090978627, 81.02669011954252, -1.143549734178217, 8.795135947180517, -1.259380021319656, 9.315182299655838, 39.00899579074751, -23.84856728439969, -39.83775699360531, 17.80425058894708, 24.03467573441748, -1.596787037004592, -5.977871388921084, -1.32541317105143 };
    Eigen::Vector2d img_size = { 1280, 720 };

    int cx = 10;
    int cy = 10;
    double sx = 0.1;
    double sy = 0.1;
    std::vector<cv::Point3f> obj_pts;
    for (int j = 0; j < cy; ++j)
    {
        for (int i = 0; i < cx; ++i)
        {
            obj_pts.push_back({ float(i * sx), float(j * sy) , 0.0f});
        }
    }

    int num_table_count = 100;
    std::vector<std::vector<cv::Point3f>> object_pts;
    std::vector<std::vector<cv::Point2f>> img_pts;
    int num_total_pts = 0;
    cv::Mat tmp_img(img_size.y(), img_size.x(), CV_8UC1);
    std::vector<Eigen::Vector3d> orig_ts;
    std::vector<Eigen::Matrix3d> orig_R;
    std::vector<Eigen::Vector3d> orig_euler;
    for (int i = 0; i < num_table_count; ++i)
    {
        tmp_img.setTo(0);

        std::vector<cv::Point2f> imgpts;
        std::vector<cv::Point3f> objpts;
        ///generate rotation translation
        double rx = (rand() / (double)RAND_MAX - 0.5) * 3.141592 / 2.0;
        double ry = (rand() / (double)RAND_MAX - 0.5) * 3.141592 / 3.0;
        double rz = (rand() / (double)RAND_MAX - 0.5) * 3.141592 / 2.0;

        double tx = (rand() / (double)RAND_MAX - 0.5) * 5;
        double ty = (rand() / (double)RAND_MAX - 0.5) * 5;
        double tz = rand() / (double)RAND_MAX * (mei ? 1.5f : 5.0f) + 0.0;
        if ( i > num_table_count - 50)
            tz = rand() / (double)RAND_MAX * 6.0 + 0.01;

        Eigen::Matrix3d m;
        m = Eigen::AngleAxisd(rz, Eigen::Vector3d::UnitZ())*
            Eigen::AngleAxisd(ry, Eigen::Vector3d::UnitY())*
            Eigen::AngleAxisd(rx, Eigen::Vector3d::UnitX());
        
        for (auto & o : obj_pts)
        {
            Eigen::Vector3d transformed = m.transpose() * Eigen::Vector3d(o.x, o.y, o.z) + 
                Eigen::Vector3d(tx, ty, tz);
            if (transformed.z() > 0)
            {
                auto ip = WorldToPlane(transformed, poly_inv, principal_point);
                if (ip.x() > 30 &&
                    ip.y() > 30 &&
                    ip.x() < img_size.x() - 30 &&
                    ip.y() < img_size.y() - 30)
                {
                    cv::circle(tmp_img, cv::Point(ip.x(), ip.y()), 3, cv::Scalar(255), -1);
                    imgpts.push_back(cv::Point2f(ip.x(), ip.y()));
                    objpts.push_back(o);
                }
            }
        }
        if (imgpts.size() > 25)
        {
            num_total_pts += (int)imgpts.size();
            img_pts.push_back(std::move(imgpts));
            object_pts.push_back(std::move(objpts));
            cv::imshow("tmp_img", tmp_img);
            cv::waitKey(0);
            orig_ts.push_back(Eigen::Vector3d(tx, ty, tz));
            orig_R.push_back(m);
            orig_euler.push_back(Eigen::Vector3d(rx, ry, rz));
        }
    }

    ///calculate table



    cv::Mat K, xi, D;
    xi.setTo(0);
    std::vector<int> indices;
    std::vector<cv::Mat> rvecs, tvecs;
    int flags = 0;
    cv::TermCriteria critia(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 2000, 0.00001);
    double rms;
    if (mei)
    {
        rms = cv::omnidir::calibrate(object_pts, img_pts,
            cv::Size(img_size.x(), img_size.y()),
            K, xi, D, rvecs, tvecs, flags, critia, indices);
    }
    else
    {
        rms = cv::calibrateCamera(object_pts, img_pts,
            cv::Size(img_size.x(), img_size.y()),
            K, D, rvecs, tvecs, flags, critia);
    }


    std::cout << K << std::endl;
    if (mei)
        std::cout << xi << std::endl;
    std::cout << D << std::endl;
    printf("rms: %f, num_total_pts: %d\n", rms, num_total_pts);

    double sum_error_t = 0;
    double sum_error_rad = 0;

    if (!mei)
    {
        indices.resize(img_pts.size());
        for (size_t i = 0; i < img_pts.size(); ++i)
            indices[i] = int(i);
    }

    for (size_t i = 0; i < indices.size(); ++i)
    {
        int j = indices[i];
        Eigen::Vector3d t(tvecs[i].at<double>(0), tvecs[i].at<double>(1), tvecs[i].at<double>(2));
        cv::Mat R;
        Eigen::Matrix3d R_eigen;
        cv::Rodrigues(rvecs[i], R);
        cv::cv2eigen(R, R_eigen);
        double tmp_err = (orig_ts[j] - t).norm();
        sum_error_t += tmp_err;
        if (tmp_err > 1)
        {
            std::cout << "Original:" << orig_ts[j] << std::endl;
            std::cout << "Calculated: " << t << std::endl;
            cv::waitKey(0);
        }

        Eigen::Vector3d r = R_eigen.eulerAngles(0, 1, 2);
        Eigen::Vector3d r_orig = orig_R[j].transpose().eulerAngles(0, 1, 2);
        tmp_err = (r - r_orig).norm();
        sum_error_rad += tmp_err;
        if (tmp_err > 3.141592 / 180.0f)
        {
            std::cout << "Original:" << orig_ts[j] << std::endl;
            std::cout << "Calculated: " << t << std::endl;
            cv::waitKey(0);
        }
    }

    printf("Num tables total: %llu \n", indices.size());
    printf("sum trans error in meter: %f, avg_err: %f \n",
        sum_error_t, sum_error_t / indices.size());
    printf("sum rad error in degree: %f, avg_err: %f\n", 
        sum_error_rad * 180.0 / 3.141592, sum_error_rad * 180.0 / 3.141592 / indices.size());
    //for (auto &)
    {
        //write result to file
        cv::FileStorage fs2(out_file_name, cv::FileStorage::WRITE);
        fs2 << "model_type" << "MEI";
        fs2 << "camera_name" << "M_WIDE_L";
        fs2 << "image_width" << (int)img_size.x();
        fs2 << "image_height" << (int)img_size.y();
        fs2 << "mirror_parameters" << "{" << "xi" << (mei ? xi.at<double>(0) : 0) << "}";

        fs2 << "distortion_parameters";fs2 << "{" <<  "k1" << D.at<double>(0)
            << "k2" << D.at<double>(1)
            << "p1" << D.at<double>(2)
            << "p2" << D.at<double>(3) << "}";

        fs2 << "projection_parameters" << "{" << "gamma1" << K.at<double>(0, 0)
            << "gamma2" << K.at<double>(1, 1)
            << "u0" << K.at<double>(0, 2)
            << "v0" << K.at<double>(1, 2) << "}";
    }

    return 0;
}

