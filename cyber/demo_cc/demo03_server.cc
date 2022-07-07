/*
    需求：客户端提交两个数字到服务端，服务端解析数字求和并将结果响应回客户端
    实现：
        1 包含头文件
        2 初始化cyber框架
        3 创建节点
        4 创建服务端
        5 处理逻辑
        6 等待关闭，释放资源
*/

#include "cyber/cyber.h"
#include "cyber/demo_base_proto/addints.pb.h"

using apollo::cyber::demo_base_proto::AddInts_Request;
using apollo::cyber::demo_base_proto::AddInts_Response;

void cb(const std::shared_ptr<AddInts_Request>& request, const std::shared_ptr<AddInts_Response>& response)
{
    int64_t num1 = request->num1();
    int64_t num2 = request->num2();

    int64_t sum = num1 + num2;
    response->set_sum(sum);
}
int main(int argc, char const *argv[])
{
    /* code */
    // 2 初始化cyber框架
    apollo::cyber::Init(argv[0]);
    // 3 创建节点
    auto server_node = apollo::cyber::CreateNode("addIntsServer_node");
    // 4 创建服务端
    auto server = server_node->CreateService<AddInts_Request, AddInts_Response>("addints", cb);
    apollo::cyber::WaitForShutdown();
    return 0;
}
