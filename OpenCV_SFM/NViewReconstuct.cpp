#include <iostream>
#include <vector>
#include <io.h>
#include <fstream>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

#include "ceres/ceres.h" 
#include "ceres/rotation.h"
#include "glog/logging.h" 


using namespace std;
using namespace cv;

// --------------------
// void getAllFiles(string path, vector<string>& files, string format);

void init_structure(
	const Mat& K,
	const vector<vector<KeyPoint>>& key_points_for_all,
	const vector<vector<Vec3b>>& colors_for_all,
	const vector<vector<DMatch>>& matches_for_all,
	vector<Point3d>& structure,
	vector<vector<int>>& correspond_struct_idx,
	vector<Vec3b>& colors,
	vector<Mat>& rotations,
	vector<Mat>& motions
);
void get_matched_points(
	const vector<KeyPoint>& p1,
	const vector<KeyPoint>& p2,
	const vector<DMatch> matches,
	vector<Point2f>& out_p1,
	vector<Point2f>& out_p2);
void get_matched_colors(
	const vector<Vec3b>& c1,
	const vector<Vec3b>& c2,
	const vector<DMatch> matches,
	vector<Vec3b>& out_c1,
	vector<Vec3b>& out_c2
);
bool find_transform(const Mat& K, const vector<Point2f>& p1, const vector<Point2f>& p2, Mat& R, Mat& T, Mat& mask);

void maskout_points(const Mat& mask, vector<Point2f>& p1);
void maskout_2d_pts_pair(const Mat& mask, vector<Point2f>& pts1, vector<Point2f>& pts2);

void maskout_colors(const Mat& mask, vector<Vec3b>& p1);

void bundle_adjustment(
	Mat& intrinsic,
	vector<Mat>& extrinsics,
	vector<vector<int>>& correspond_struct_idx,
	vector<vector<KeyPoint>>& key_points_for_all,
	vector<Point3d>& structure
);

void reconstruct(const Mat& K,
	Mat& R1, Mat& T1, Mat& R2, Mat& T2,
	vector<Point2f>& p1, vector<Point2f>& p2,
	vector<Point3d>& structure);
void get_objpoints_and_imgpoints(
	vector<DMatch>& matches,
	vector<int>& struct_indices,
	vector<Point3d>& structure,
	vector<KeyPoint>& key_points,
	vector<Point3f>& object_points,
	vector<Point2f>& image_points);
void fusion_structure(
	vector<DMatch>& matches,
	vector<int>& struct_indices,
	vector<int>& next_struct_indices,
	vector<Point3d>& structure,
	vector<Point3d>& next_structure,
	vector<Vec3b>& colors,
	vector<Vec3b>& next_colors);
void extract_features(
	vector<string>& image_names,
	vector<vector<KeyPoint>>& key_points_for_all,
	vector<Mat>& descriptor_for_all,
	vector <vector<Vec3b>>& colors_for_all
);
void match_features(const Mat& query, const Mat& train, vector<DMatch>& matches);
void match_features(const vector<Mat>& descriptor_for_all, vector<vector<DMatch>>& matches_for_all);
void save_structure(string file_name,
	vector<Mat>& rotations, 
	vector<Mat>& motions, 
	vector<Point3d>& structure,
	vector<Vec3b>& colors);

// --------------------
struct ReprojectCost
{
	cv::Point2d observation;

	ReprojectCost(cv::Point2d& observation)
		: observation(observation)
	{
	}

	template <typename T>
	bool operator()(const T* const intrinsic, const T* const extrinsic, const T* const pos3d, T* residuals) const
	{
		const T* r = extrinsic;
		const T* t = &extrinsic[3];

		T pos_proj[3];
		ceres::AngleAxisRotatePoint(r, pos3d, pos_proj);

		// Apply the camera translation
		pos_proj[0] += t[0];
		pos_proj[1] += t[1];
		pos_proj[2] += t[2];

		const T x = pos_proj[0] / pos_proj[2];
		const T y = pos_proj[1] / pos_proj[2];

		const T fx = intrinsic[0];
		const T fy = intrinsic[1];
		const T cx = intrinsic[2];
		const T cy = intrinsic[3];

		// Apply intrinsic
		const T u = fx * x + cx;
		const T v = fy * y + cy;

		residuals[0] = u - T(observation.x);
		residuals[1] = v - T(observation.y);

		return true;
	}
};

// --------------------

const string dir = "../Images/";

int main(int argc, char** argv)
{
	vector<string> img_names;
	img_names.push_back(dir + "0000.png");
	img_names.push_back(dir + "0001.png");
	img_names.push_back(dir + "0002.png");
	img_names.push_back(dir + "0003.png");
	img_names.push_back(dir + "0004.png");
	img_names.push_back(dir + "0005.png");
	img_names.push_back(dir + "0006.png");
	img_names.push_back(dir + "0007.png");
	img_names.push_back(dir + "0008.png");
	img_names.push_back(dir + "0009.png");
	img_names.push_back(dir + "0010.png");

	// 相机内参家矩阵
	Mat K(Matx33d(
		2759.48, 0, 1520.69,
		0, 2764.16, 1006.81,
		0, 0, 1));

	vector<vector<cv::KeyPoint>> key_points_for_all;
	vector<Mat> descriptor_for_all;
	vector<vector<Vec3b>> colors_for_all;
	vector<vector<DMatch>> matches_for_all;

	// 提取所有图像的特征
	extract_features(img_names, key_points_for_all, descriptor_for_all, colors_for_all);

	// 对所有图像进行顺次的特征匹配
	match_features(descriptor_for_all, matches_for_all);

	vector<Point3d> structure;
	vector<vector<int>> correspond_struct_idx;	// 保存第i副图像中第j特征点对应的structure中点的索引
	vector<Vec3b> colors;  // 3个字节表示一个RGB或BGR颜色
	vector<Mat> rotations;
	vector<Mat> motions;

	// 初始化结构（三维点云） 前两帧
	init_structure(
		K,
		key_points_for_all,
		colors_for_all,
		matches_for_all,
		structure,
		correspond_struct_idx,
		colors,
		rotations,
		motions
	);

	// 增量方式重建剩余的图像
	printf("\nIncremental SFM...\n");
	for (int i = 1; i < matches_for_all.size(); ++i)
	{
		vector<Point3f> object_points;
		vector<Point2f> image_points;
		Mat r, R, T;

		// 获取第i副图像中匹配点对应的三维点，以及在第i+1副图像中对应的像素点
		get_objpoints_and_imgpoints(
			matches_for_all[i],
			correspond_struct_idx[i],
			structure,
			key_points_for_all[i + 1],
			object_points,
			image_points
		);

		// 求解变换矩阵
		cv::solvePnPRansac(object_points, image_points, K, noArray(), r, T);

		// 将旋转向量转换为旋转矩阵
		cv::Rodrigues(r, R);

		// 保存变换矩阵
		rotations.push_back(R);
		motions.push_back(T);

		vector<Point2f> p1, p2;
		vector<Vec3b> c1, c2;
		get_matched_points(key_points_for_all[i], key_points_for_all[i + 1], matches_for_all[i], p1, p2);
		get_matched_colors(colors_for_all[i], colors_for_all[i + 1], matches_for_all[i], c1, c2);

		// 根据之前求得的R, T进行三维重建
		vector<Point3d> next_structure;
		reconstruct(K, rotations[i], motions[i], R, T, p1, p2, next_structure);
		printf("Frame %d reconstructed.\n", i);

		//将新的重建结果与之前的融合
		fusion_structure(
			matches_for_all[i],
			correspond_struct_idx[i],
			correspond_struct_idx[i + 1],
			structure,
			next_structure,
			colors,
			c1
		);
		printf("Frame %d point cloud fused.\n", i);
	}

	// 保存
	save_structure("../Viewer/structure.yml", rotations, motions, structure, colors);
	cout << "Save structure done." << endl;
	getchar();
	
	printf("Start bundle adjustment fo SFM...");
	Mat intrinsic(Matx41d(K.at<double>(0, 0), K.at<double>(1, 1), K.at<double>(0, 2), K.at<double>(1, 2)));
	vector<Mat> extrinsics;
	for (size_t i = 0; i < rotations.size(); ++i)
	{
		Mat extrinsic(6, 1, CV_64FC1);
		Mat r;
		Rodrigues(rotations[i], r);

		r.copyTo(extrinsic.rowRange(0, 3));
		motions[i].copyTo(extrinsic.rowRange(3, 6));

		extrinsics.push_back(extrinsic);
	}

	bundle_adjustment(intrinsic, extrinsics, correspond_struct_idx, key_points_for_all, structure);

	return 0;
}

void save_structure(string file_name,
	vector<Mat>& rotations,
	vector<Mat>& motions,
	vector<Point3d>& structure,
	vector<Vec3b>& colors)
{
	int n = (int)rotations.size();

	FileStorage fs(file_name, FileStorage::WRITE);
	fs << "Camera Count" << n;
	fs << "Point Count" << (int)structure.size();

	fs << "Rotations" << "[";
	for (size_t i = 0; i < n; ++i)
	{
		fs << rotations[i];
	}
	fs << "]";

	fs << "Motions" << "[";
	for (size_t i = 0; i < n; i++)
	{
		fs << motions[i];
	}
	fs << "]";

	fs << "Points" << "[";
	for (size_t i = 0; i < structure.size(); ++i)
	{
		fs << structure[i];
	}
	fs << "]";

	fs << "Colors" << "[";
	for (size_t i = 0; i < colors.size(); ++i)
	{
		fs << colors[i];
	}
	fs << "]";

	fs.release();

}

void extract_features(
	vector<string>& image_names,
	vector<vector<KeyPoint>>& key_points_for_all,
	vector<Mat>& descriptor_for_all,
	vector <vector<Vec3b>>& colors_for_all
)
{
	key_points_for_all.clear();
	descriptor_for_all.clear();
	Mat img;

	// 读取图像，获取图像特征点并保存
	Ptr<Feature2D> sift = cv::SIFT::create();
	for (auto it = image_names.begin(); it != image_names.end(); ++it)
	{
		// 读取图像
		img = imread(*it);
		if (img.empty())
		{
			continue;
		}
		cout << "Extracting features: " << *it << endl;

		vector<KeyPoint> key_points;
		Mat descriptor;

		// 偶尔出现内存分配失败的错误  Detects keypoints and computes the descriptors
		// sift->detectAndCompute(image, noArray(), key_points, descriptor);
		sift->detect(img, key_points);
		sift->compute(img, key_points, descriptor);

		// 特征点过少，则排除该图像
		if (key_points.size() <= 10)
		{
			continue;
		}

		key_points_for_all.push_back(key_points);
		descriptor_for_all.push_back(descriptor);

		// 三通道 存放该位置三通道颜色
		vector<Vec3b> colors(key_points.size());
		for (int i = 0; i < key_points.size(); ++i)
		{
			Point2f& kp = key_points[i].pt;
			const int& y = int(kp.y);
			const int& x = int(kp.x);
			if (y <= img.rows && x <= img.cols)
			{
				colors[i] = img.at<Vec3b>(y, x);
			}
			else
			{
				printf("[Warning]: pt2d[%.3f, %.3f] out of image range.\n", kp.x, kp.y);
			}
		}

		colors_for_all.push_back(colors);
	}
}

void match_features(const vector<Mat>& descriptor_for_all, vector<vector<DMatch>>& matches_for_all)
{
	matches_for_all.clear();
	// n个图像，两两顺次有 n-1 对匹配
	// 1与2匹配，2与3匹配，3与4匹配，以此类推
	for (int i = 0; i < descriptor_for_all.size() - 1; ++i)
	{
		cout << "Matching images " << i << " - " << i + 1 << endl;
		vector<DMatch> matches;
		match_features(descriptor_for_all[i], descriptor_for_all[i + 1], matches);
		matches_for_all.push_back(matches);
	}
}

void match_features(const Mat& query, const Mat& train, vector<DMatch>& matches)
{
	vector<vector<DMatch>> knn_matches;
	BFMatcher matcher(NORM_L2);
	matcher.knnMatch(query, train, knn_matches, 2);  // 取top2匹配最好的匹配

	// 获取满足Ratio Test的最小匹配的距离
	float min_dist = FLT_MAX;
	for (int i = 0; i < knn_matches.size(); ++i)
	{
		// Rotio Test
		if (knn_matches[i][0].distance > 0.6 * knn_matches[i][1].distance)
		{
			continue;
		}

		float dist = knn_matches[i][0].distance;
		if (dist < min_dist)
		{
			min_dist = dist;
		}
	}

	matches.clear();
	for (size_t i = 0; i < knn_matches.size(); ++i)
	{
		// 排除不满足Ratio Test的点和匹配距离过大的点
		if (
			knn_matches[i][0].distance > 0.6 * knn_matches[i][1].distance ||
			knn_matches[i][0].distance > 5 * max(min_dist, 10.0f)
			)
		{
			continue;
		}

		// 保存匹配点
		matches.push_back(knn_matches[i][0]);
	}
}

void init_structure(
	const Mat& K,
	const vector<vector<KeyPoint>>& key_points_for_all,
	const vector<vector<Vec3b>>& colors_for_all,
	const vector<vector<DMatch>>& matches_for_all,
	vector<Point3d>& structure,
	vector<vector<int>>& correspond_struct_idx,
	vector<Vec3b>& colors,
	vector<Mat>& rotations,
	vector<Mat>& motions
)
{
	// 计算前两帧(编号0和1)图像之间的变换矩阵
	vector<Point2f> p1, p2;
	vector<Vec3b> c2;
	Mat R, T;	// 旋转矩阵和平移向量
	Mat mask;	// mask中大于零的点代表匹配点，等于零的点代表失配点
	get_matched_points(key_points_for_all[0], key_points_for_all[1], matches_for_all[0], p1, p2);
	get_matched_colors(colors_for_all[0], colors_for_all[1], matches_for_all[0], colors, c2);

	// 根据匹配的特征点计算本质矩阵并分解得到R,T矩阵
	find_transform(K, p1, p2, R, T, mask);
	//printf("mask type: %d\n", mask.type());  // mask: CV_8U(uint8)

	// 对头两幅图像进行三维重建
	//maskout_points(mask, p1);
	//maskout_points(mask, p2);
	maskout_2d_pts_pair(mask, p1, p2);
	maskout_colors(mask, colors);

	Mat R0 = Mat::eye(3, 3, CV_64FC1);
	Mat T0 = Mat::zeros(3, 1, CV_64FC1);
	reconstruct(K, R0, T0, R, T, p1, p2, structure);  // 三角化

	// 保存变换矩阵
	rotations = { R0, R };
	motions = { T0, T };

	// 将correspond_struct_idx的大小初始化为与key_points_for_all完全一致
	correspond_struct_idx.clear();
	correspond_struct_idx.resize(key_points_for_all.size());
	for (int i = 0; i < key_points_for_all.size(); ++i)
	{
		correspond_struct_idx[i].resize(key_points_for_all[i].size(), -1);
	}

	// 填写头两幅图像的结构索引
	int idx = 0;
	const vector<DMatch>& matches = matches_for_all[0];
	for (int i = 0; i < matches.size(); ++i)
	{
		if (mask.at<uchar>(i) == 0)
		{
			continue;
		}

		correspond_struct_idx[0][matches[i].queryIdx] = idx;	// 如果两个点对应的idx 相等 表明它们是同一特征点 idx 就是structure中对应的空间点坐标索引
		correspond_struct_idx[1][matches[i].trainIdx] = idx;
		++idx;
	}
}

void get_matched_points(
	const vector<KeyPoint>& p1,
	const vector<KeyPoint>& p2,
	const vector<DMatch> matches,
	vector<Point2f>& out_p1,
	vector<Point2f>& out_p2)
{
	out_p1.clear();
	out_p2.clear();
	for (int i = 0; i < matches.size(); ++i)
	{
		out_p1.push_back(p1[matches[i].queryIdx].pt);
		out_p2.push_back(p2[matches[i].trainIdx].pt);
	}
}

void get_matched_colors(
	const vector<Vec3b>& c1,
	const vector<Vec3b>& c2,
	const vector<DMatch> matches,
	vector<Vec3b>& out_c1,
	vector<Vec3b>& out_c2
)
{
	out_c1.clear();
	out_c2.clear();
	for (int i = 0; i < matches.size(); ++i)
	{
		out_c1.push_back(c1[matches[i].queryIdx]);
		out_c2.push_back(c2[matches[i].trainIdx]);
	}
}

bool find_transform(const Mat& K,
	const vector<Point2f>& p1, const vector<Point2f>& p2,
	Mat& R, Mat& T,
	Mat& mask)
{
	// 根据内参数矩阵获取相机的焦距和光心坐标(主点坐标)
	double focal_length = 0.5 * (K.at<double>(0) + K.at<double>(4));
	Point2d principle_point(K.at<double>(2), K.at<double>(5));

	// 根据匹配点求取本征矩阵，使用RANSAC，进一步排除失配点
	const Mat& E = findEssentialMat(p1, p2, focal_length, principle_point, RANSAC, 0.999, 1.0, mask);
	if (E.empty())
	{
		return false;
	}

	double feasible_count = countNonZero(mask);	// 得到非零元素，即数组中的有效点
	// cout << (int)feasible_count << " - in - " << p1.size() << endl;

	// 对于RANSAC而言，outlier数量大于50%时，结果是不可靠的
	if (feasible_count <= 15 || (feasible_count / p1.size()) < 0.6)
	{
		return false;
	}

	// 分解本征矩阵，获取相对变换
	int pass_count = cv::recoverPose(E, p1, p2, R, T, focal_length, principle_point, mask);

	// cout << "pass_count = " << pass_count << endl;

	// 同时位于两个相机前方的点的数量要足够大
	if (((double)pass_count) / feasible_count < 0.7)
	{
		return false;
	}
	return true;
}

void maskout_points(const Mat& mask, vector<Point2f>& p1)
{
	vector<Point2f> p1_copy = p1;  // 此处并没有使用左值引用&, 故并不指向同一块内存地址
	p1.clear();

	for (int i = 0; i < mask.rows; ++i)
	{
		if (mask.at<uchar>(i) > 0)
		{
			p1.push_back(p1_copy[i]);
		}
	}
}

void maskout_2d_pts_pair(const Mat& mask, vector<Point2f>& pts1, vector<Point2f>& pts2)
{
	vector<Point2f> pts1_copy = pts1;
	vector<Point2f> pts2_copy = pts2;

	pts1.clear();
	pts2.clear();
	pts1.reserve(pts1_copy.size());  // 预先分配好内存，避免push_back阶段多次分配内存
	pts2.reserve(pts2_copy.size());

	for (int i = 0; i < mask.rows; i++)
	{
		if (mask.at<uchar>(i) > 0)
		{
			pts1.push_back(pts1_copy[i]);
			pts2.push_back(pts2_copy[i]);
		}
	}
}

void maskout_colors(const Mat& mask, vector<Vec3b>& pts_color)
{
	vector<Vec3b> p1_copy = pts_color;
	pts_color.clear();
	pts_color.reserve(p1_copy.size());

	for (int i = 0; i < mask.rows; ++i)
	{
		if (mask.at<uchar>(i) > 0)
		{
			pts_color.push_back(p1_copy[i]);
		}
	}
}

void reconstruct(const Mat& K,
	Mat& R1, Mat& T1, Mat& R2, Mat& T2,
	vector<Point2f>& p1, vector<Point2f>& p2,
	vector<Point3d>& structure)
{
	// 两个相机的投影矩阵[R, T], triangulatePoints只支持float型
	Mat proj1(3, 4, CV_32FC1);
	Mat proj2(3, 4, CV_32FC1);

	R1.convertTo(proj1(Range(0, 3), Range(0, 3)), CV_32FC1);
	//T1.convertTo(proj2(Range(0, 3), Range(3, 4)), CV_32FC1);
	T1.convertTo(proj1.col(3), CV_32FC1);

	R2.convertTo(proj2(Range(0, 3), Range(0, 3)), CV_32FC1);
	//T2.convertTo(proj2(Range(0, 3), Range(3, 4)), CV_32FC1);
	T2.convertTo(proj2.col(3), CV_32FC1);

	Mat fK;
	K.convertTo(fK, CV_32FC1);
	proj1 = fK * proj1;
	proj2 = fK * proj2;

	// 三角测量重建3D坐标
	cv::Mat pts4d;
	cv::triangulatePoints(proj1, proj2, p1, p2, pts4d);

	structure.clear();
	structure.reserve(pts4d.cols);
	for (int i = 0; i < pts4d.cols; ++i)
	{
		Mat_<float> col = pts4d.col(i);
		col /= col(3);	// 齐次坐标
		structure.push_back(Point3f(col(0), col(1), col(2)));
	}
}

void bundle_adjustment(
	Mat& intrinsic,
	vector<Mat>& extrinsics,
	vector<vector<int>>& correspond_struct_idx,
	vector<vector<KeyPoint>>& key_points_for_all,
	vector<Point3d>& structure
)
{
	ceres::Problem problem;

	// load extrinsics (rotations and motions)
	for (size_t i = 0; i < extrinsics.size(); ++i)
	{
		problem.AddParameterBlock(extrinsics[i].ptr<double>(), 6);
	}

	// fix the first camera.
	problem.SetParameterBlockConstant(extrinsics[0].ptr<double>());

	// load intrinsic
	problem.AddParameterBlock(intrinsic.ptr<double>(), 4); // fx, fy, cx, cy

	// loss function make bundle adjustment robuster.
	ceres::LossFunction* loss_function = new ceres::HuberLoss(4);  

	// load points
	for (size_t img_idx = 0; img_idx < correspond_struct_idx.size(); ++img_idx)
	{
		vector<int>& point3d_ids = correspond_struct_idx[img_idx];
		vector<KeyPoint>& key_points = key_points_for_all[img_idx];
		for (size_t point_idx = 0; point_idx < point3d_ids.size(); ++point_idx)
		{
			int point3d_id = point3d_ids[point_idx];
			if (point3d_id < 0)
				continue;

			Point2d observed = key_points[point_idx].pt;

			// 模板参数中，第一个为代价函数的类型，第二个为代价的维度，剩下三个分别为代价函数第一第二还有第三个参数的维度
			ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<ReprojectCost, 2, 4, 6, 3>(new ReprojectCost(observed));

			problem.AddResidualBlock(
				cost_function,
				loss_function,
				intrinsic.ptr<double>(),			// Intrinsic
				extrinsics[img_idx].ptr<double>(),	// View Rotation and Translation
				&(structure[point3d_id].x)			// Point in 3D space
			);
		}
	}

	// Solve BA
	ceres::Solver::Options ceres_config_options;
	ceres_config_options.minimizer_progress_to_stdout = false;
	ceres_config_options.logging_type = ceres::SILENT;
	ceres_config_options.num_threads = 1;
	ceres_config_options.preconditioner_type = ceres::JACOBI;
	ceres_config_options.linear_solver_type = ceres::SPARSE_SCHUR;
	ceres_config_options.sparse_linear_algebra_library_type = ceres::EIGEN_SPARSE;

	ceres::Solver::Summary summary;
	ceres::Solve(ceres_config_options, &problem, &summary);

	if (!summary.IsSolutionUsable())
	{
		std::cout << "Bundle Adjustment failed." << std::endl;
	}
	else
	{
		// Display statistics about the minimization
		std::cout << std::endl
			<< "Bundle Adjustment statistics (approximated RMSE):\n"
			<< " #views: " << extrinsics.size() << "\n"
			<< " #residuals: " << summary.num_residuals << "\n"
			<< " Initial RMSE: " << std::sqrt(summary.initial_cost / summary.num_residuals) << "\n"
			<< " Final RMSE: " << std::sqrt(summary.final_cost / summary.num_residuals) << "\n"
			<< " Time (s): " << summary.total_time_in_seconds << "\n"
			<< std::endl;
	}
}

void get_objpoints_and_imgpoints(
	vector<DMatch>& matches,
	vector<int>& struct_indices,
	vector<Point3d>& structure,
	vector<KeyPoint>& key_points,
	vector<Point3f>& object_points,
	vector<Point2f>& image_points)
{
	object_points.clear();
	image_points.clear();

	for (int i = 0; i < matches.size(); ++i)
	{
		int query_idx = matches[i].queryIdx;
		int train_idx = matches[i].trainIdx;

		int struct_idx = struct_indices[query_idx];
		if (struct_idx < 0)	// 表明跟前一副图像没有匹配点
		{
			continue;
		}

		object_points.push_back(structure[struct_idx]);
		image_points.push_back(key_points[train_idx].pt);  // train中对应关键点的坐标 二维
	}
}

void fusion_structure(
	vector<DMatch>& matches,
	vector<int>& struct_indices,
	vector<int>& next_struct_indices,
	vector<Point3d>& structure,
	vector<Point3d>& next_structure,
	vector<Vec3b>& colors,
	vector<Vec3b>& next_colors)
{
	for (int i = 0; i < matches.size(); ++i)
	{
		int query_idx = matches[i].queryIdx;
		int train_idx = matches[i].trainIdx;

		int struct_idx = struct_indices[query_idx];
		if (struct_idx >= 0)	// 若该点在空间中已经存在，则这对匹配点对应的空间点应该是同一个，索引要相同
		{
			next_struct_indices[train_idx] = struct_idx;
			continue;
		}

		// 若该点在空间中未存在，将该点加入到结构中，且这对匹配点的空间索引都为新加入的点的索引
		structure.push_back(next_structure[i]);
		colors.push_back(next_colors[i]);
		struct_indices[query_idx] = next_struct_indices[train_idx] = int(structure.size()) - 1;
	}
}

//获取特定格式的文件名    
//void getAllFiles(string path, vector<string>& files, string format)
//{
//	long  hFile = 0;//文件句柄  64位下long 改为 intptr_t
//	struct _finddata_t fileinfo;//文件信息 
//	string p;
//	if ((hFile = _findfirst(p.assign(path).append("\\*" + format).c_str(), &fileinfo)) != -1) //文件存在
//	{
//		do
//		{
//			if ((fileinfo.attrib & _A_SUBDIR))//判断是否为文件夹
//			{
//				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)//文件夹名中不含"."和".."
//				{
//					files.push_back(p.assign(path).append("\\").append(fileinfo.name)); //保存文件夹名
//					getAllFiles(p.assign(path).append("\\").append(fileinfo.name), files, format); //递归遍历文件夹
//				}
//			}
//			else
//			{
//				files.push_back(p.assign(path).append("\\").append(fileinfo.name));//如果不是文件夹，储存文件名
//			}
//		} while (_findnext(hFile, &fileinfo) == 0);
//		_findclose(hFile);
//	}
//}