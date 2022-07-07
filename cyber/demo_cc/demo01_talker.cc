/*
    需求：按照某个频率循环发布学生信息
    实现：
        1. 包含头文件
        2. 初始化cyber框架
        3. 创建节点
        4. 创建发布者
        5. 组织并发布数据
        6. 节点关闭，释放资源

*/
#include "cyber/cyber.h"
#include "cyber/demo_base_proto/student.pb.h"

using apollo::cyber::demo_base_proto::Student;

int main(int argc, char const *argv[])
{
    /* code */
    // 2. 初始化cyber框架
    apollo::cyber::Init(argv[0]);
    // 3. 创建节点
    auto talker_node = apollo::cyber::CreateNode("ergou");
    // 4. 创建发布者
    auto talker = talker_node->CreateWriter<Student>("chatter");
    // 5. 组织并发布数据
    apollo::cyber::Rate rate(0.5);
    uint64_t seq = 0;
    while(apollo::cyber::OK())
    {
        seq++;
        AINFO << "the " << seq << "th data";
        auto stu_ptr = std::make_shared<Student>();
        stu_ptr->set_name("huluwa");
        stu_ptr->set_age(seq);
        stu_ptr->set_height(2);
        stu_ptr->add_books("shuxue");
        stu_ptr->add_books("yuwen");
        stu_ptr->add_books("yingyu");
        talker->Write(stu_ptr);
        rate.Sleep();
    }
    // 6. 节点关闭，释放资源
    apollo::cyber::WaitForShutdown();
    return 0;
}
