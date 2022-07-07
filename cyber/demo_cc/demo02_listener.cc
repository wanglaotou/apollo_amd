/*
    需求：发布学生相关信息
    实现：
        1. 包含头文件
        2. cyber初始化
        3. 创建节点
        4. 创建订阅者
        5. 解析数据
        6. 节点关闭，释放资源
*/

#include "cyber/cyber.h"
#include "cyber/demo_base_proto/student.pb.h"

using apollo::cyber::demo_base_proto::Student;

void cb(const std::shared_ptr<Student>& stu)
{
    AINFO << "name: " << stu->name();
    AINFO << "age: " << stu->age();
    AINFO << "height: " << stu->height();
    for(int i=0;i<stu->books_size();i++)
    {
        AINFO << "book: "<< stu->books(i);
    }
    AINFO << "-------------------------------------";
}
int main(int argc, char const *argv[])
{
    /* code */
    apollo::cyber::Init(argv[0]);
    AINFO << "订阅方创建 .......";
    auto listener_node = apollo::cyber::CreateNode("cuihua");
    auto listener = listener_node->CreateReader<Student>("chatter", cb);
    apollo::cyber::WaitForShutdown();
    return 0;
}
