
#include "cyber/cyber.h"
#include "cyber/parameter/parameter_server.h"

using apollo::cyber::ParameterServer;
using apollo::cyber::Parameter;

int main(int argc, char const *argv[])
{
    /* code */
    // 初始化cyber框架
    apollo::cyber::Init(argv[0]);
    // 创建节点
    std::shared_ptr<apollo::cyber::Node> server_node = apollo::cyber::CreateNode("car_param");
    // 创建参数服务器
    auto server = std::make_shared<ParameterServer>(server_node);

    // 5 操作参数（增，查，改）
    // 5-1增
    server->SetParameter(Parameter("car_type", "apollo"));
    server->SetParameter(Parameter("height", 1.65));
    server->SetParameter(Parameter("laser", 4));

    // 5-2查
    // a. 获取某个指定参数
    Parameter temp;
    server->GetParameter("car_type",&temp);
    AINFO <<temp.Name()<<"=="<<temp.AsString();
    server->GetParameter("height",&temp);
    AINFO <<  temp.Name()<<"=="<<temp.AsDouble();
    server->GetParameter("laser",&temp);
    AINFO << temp.Name()<<"=="<<temp.AsInt64();

    // b.获取所有参数
    std::vector<Parameter> ps;
    server->ListParameters(&ps);
    for (auto &&p :  ps)
    {
        /* code */
        AINFO << p.Name() << " ======== " << p.TypeName() <<"-----------"<<p.DebugString();
    }
    
    // 5-3改
    server->SetParameter(Parameter("laser", 10));
    server->GetParameter("laser",&temp);
    AINFO <<temp.Name()<<"=="<<temp.AsInt64();

    apollo::cyber::WaitForShutdown();
    return 0;
}
