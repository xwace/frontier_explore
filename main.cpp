#include <iostream>
#include <random>
#include <utility>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

struct Frontier{
    int size;
    double min_distance;
    cv::Point travel_point;
};

std::vector<unsigned int> nhood4(unsigned int idx, const Mat& costmap)
{
    // get 4-connected neighbourhood indexes, check for edge of map
    std::vector<unsigned int> out;

    unsigned int size_x_ = costmap.cols, size_y_ = costmap.rows;

    if (idx > size_x_ * size_y_ -1)
    {
        return out;
    }

    if (idx % size_x_ > 0)
    {
        out.push_back(idx - 1);
    }
    if (idx % size_x_ < size_x_ - 1)
    {
        out.push_back(idx + 1);
    }
    if (idx >= size_x_)
    {
        out.push_back(idx - size_x_);
    }
    if (idx < size_x_*(size_y_-1))
    {
        out.push_back(idx + size_x_);
    }
    return out;
}

std::vector<unsigned int> nhood8(unsigned int idx, const Mat& costmap)
{
    // get 8-connected neighbourhood indexes, check for edge of map
    std::vector<unsigned int> out = nhood4(idx, costmap);

    unsigned int size_x_ = costmap.cols, size_y_ = costmap.rows;

    if (idx > size_x_ * size_y_ -1)
    {
        return out;
    }

    if (idx % size_x_ > 0 && idx >= size_x_)
    {
        out.push_back(idx - 1 - size_x_);
    }
    if (idx % size_x_ > 0 && idx < size_x_*(size_y_-1))
    {
        out.push_back(idx - 1 + size_x_);
    }
    if (idx % size_x_ < size_x_ - 1 && idx >= size_x_)
    {
        out.push_back(idx + 1 - size_x_);
    }
    if (idx % size_x_ < size_x_ - 1 && idx < size_x_*(size_y_-1))
    {
        out.push_back(idx + 1 + size_x_);
    }

    return out;
}

bool nearestCell(unsigned int &result, unsigned int start, unsigned char val,  // NOLINT (runtime/references)
                 const Mat& costmap)  // NOLINT (runtime/references)
{
    const unsigned char* map = costmap.data;
    const unsigned int size_x = costmap.cols, size_y = costmap.rows;

    if (start >= size_x * size_y)
    {
        return false;
    }

    // initialize breadth first search
    std::queue<unsigned int> bfs;
    std::vector<bool> visited_flag(size_x * size_y, false);

    // push initial cell
    bfs.push(start);
    visited_flag[start] = true;

    // search for neighbouring cell matching value
    while (!bfs.empty())
    {
        unsigned int idx = bfs.front();
        bfs.pop();

        // return if cell of correct value is found
        if (map[idx] == val)
        {
            result = idx;
            return true;
        }

        // iterate over all adjacent unvisited cells
        for(auto nbr:nhood8(idx, costmap))
        {
            if (!visited_flag[nbr])
            {
                bfs.push(nbr);
                visited_flag[nbr] = true;
            }
        }
    }

    return false;
}

bool isNewFrontierCell(const Mat& costmap, unsigned int idx, const std::vector<bool>& frontier_flag)
{
    const uchar* map_ = costmap.data;
    // check that cell is unknown and not already marked as frontier
    if (map_[idx] != 2 || frontier_flag[idx])
    {
        return false;
    }

    // frontier cells should have at least one cell in 4-connected neighbourhood that is free
    for(auto nbr:nhood4(idx, costmap))
    {
        if (map_[nbr] == 0)
        {
            return true;
        }
    }
    return false;
}

static Mat frontierImg;
Frontier buildNewFrontier(const Mat& img, unsigned int initial_cell, unsigned int reference,
                          std::vector<bool>& frontier_flag)
{
    // initialize frontier structure
    Frontier output;
    cv::Point centroid, middle;
    output.size = 1;
    output.min_distance = std::numeric_limits<double>::infinity();

    // push initial gridcell onto queue
    std::queue<unsigned int> bfs;
    bfs.push(initial_cell);

    double reference_x(0), reference_y(0);//当前扫地机位置

    while (!bfs.empty())
    {
        unsigned int idx = bfs.front();
        bfs.pop();

        frontierImg.at<uchar>(idx/img.cols,idx %img.cols) = 10;
        namedWindow("frontierImg",2);
        imshow("frontierImg",frontierImg*45);
        waitKey();

        // try adding cells in 8-connected neighborhood to frontier
        for(auto nbr: nhood8(idx, img))
        {
            // check if neighbour is a potential frontier cell
            if (isNewFrontierCell(img, nbr, frontier_flag))
            {
                // mark cell as frontier
                frontier_flag[nbr] = true;
                unsigned int mx, my;
                double wx, wy;
                wy = nbr / img.cols;
                wx = nbr % img.cols;

                // update frontier size
                output.size++;

                // update centroid of frontier
                centroid.x += wx;
                centroid.y += wy;

                // determine frontier's distance from robot, going by closest gridcell to robot
                double distance = cv::norm(Point(wx- reference_x, wy-reference_y));
                if (distance < output.min_distance)
                {
                    output.min_distance = distance;
                    middle.x = wx;
                    middle.y = wy;
                }

                // add to queue for breadth first search
                bfs.push(nbr);

                frontierImg.at<uchar>(wy,wx) = 10;
                namedWindow("frontierImg",2);
                imshow("frontierImg",frontierImg*45);
                waitKey();
            }
        }
    }

    // average out frontier centroid
    centroid.x /= output.size;
    centroid.y /= output.size;

    string travel_point_ = "centroid";
    if (travel_point_ == "closest")
    {
        // point already set
    }
    else if (travel_point_ == "middle")
    {
        output.travel_point = middle;
    }
    else if (travel_point_ == "centroid")
    {
        output.travel_point = centroid;
    }
    else
    {
        printf("Invalid 'frontier_travel_point' parameter, falling back to 'closest'");
        // point already set
    }

    return output;
}

int main() {
    Mat img(20,20,0,Scalar(0));
    img.col(5) = 2;
    img.col(6) = 2;
    img.col(7) = 2;
    shuffle(img.begin<uchar>(),img.end<uchar>(), std::mt19937(std::random_device()()));
    img.col(5) = 2;
    img.col(6) = 2;
    img.col(7) = 2;
    img.at<uchar>(5,5) = 0;
    img.at<uchar>(5,6) = 0;
    img.at<uchar>(5,7) = 0;
    cout<<img<<endl;

    unsigned int r{0};

    auto map_ = img.data;
    std::list<Frontier> frontier_list;
    std::vector<bool> frontier_flag(img.total(), false);
    std::vector<bool> visited_flag(img.total(), false);
    std::queue<unsigned int> bfs;

    unsigned int init_pose = 14;
    bool near = nearestCell(init_pose, 15, 0, img);
    bfs.push(init_pose);//扫地机初始点

    Mat bfsImg = img.clone();
    frontierImg = img.clone();

    while (!bfs.empty())
    {
        unsigned int idx = bfs.front();
        bfs.pop();

        // iterate over 4-connected neighbourhood
        for(auto nbr:nhood4(idx, img))
        {
            // add to queue all free, unvisited cells, use descending search in case initialized on non-free cell
            if (map_[nbr] <= map_[idx] && !visited_flag[nbr])
            {
                visited_flag[nbr] = true;
                bfs.push(nbr);

                bfsImg.at<uchar>(nbr/img.cols,nbr%img.cols) = 10;
                namedWindow("bfs",2);
                imshow("bfs",bfsImg*45);
                waitKey();
            }

            else if (isNewFrontierCell(img, nbr, frontier_flag))
            {
                frontier_flag[nbr] = true;
                Frontier new_frontier = buildNewFrontier(img, nbr, 6, frontier_flag);
                cout<<"获取了一条边界: "<<new_frontier.size<<endl;
                if (new_frontier.size > 5)//5个栅格为机身长度
                {
                    frontier_list.push_back(new_frontier);
                }
            }
        }
    }

    cout<<"size: "<<frontier_list.size()<<endl;
    cout<<"frontier_list: "<<frontier_list.front().travel_point<<endl;
}
