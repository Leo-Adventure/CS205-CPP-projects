#include <iostream>
#include <string>
#include <stack>

typedef long long ll;

using namespace std;

string addString(string, string);
int main(int argc, char **argv)
{
    bool first_time = true;//判断是否是第一次进入程序，
    //因为涉及到命令行输入的问题，只有第一次才可以通过命令行输入两个数字
loop:

    stack<char> res;
    string f, s;//代表起始输入的两个字符串
    if (argc < 3 || !first_time)
    {

        cout << "Please input two integers: (enter \" quit \" to quit)" << endl;
        cin >> f;
        //程序会一直循环进行指导用户输入quit为止
        if (f == "quit")
        {
            return 0;
        }
        cin >> s;
    }
    else
    {
        first_time = false;//通过命令行参数传入第一次运算的数字时，改变first_time的值为 false;
        for (int i = 1; i < argc; i++)
        {
            if (i == 1)
            {
                f = argv[i];
            }
            else
            {
                s = argv[i];
            }
        }
    }

    //当输入的第一个字符串为quit时，退出程序
    if (f == "quit")
    {
        return 0;
    }

    //判断结果的正负
    bool isNegative = false;
    if ((f[0] == '-' && s[0] != '-') || (f[0] != '-' && s[0] == '-'))
    {
        isNegative = true;
    }

    ll fsize = f.size();
    ll ssize = s.size();
    string first = "", second = "";

    //调整成为竖式乘法上端较大，下端较小，便于之后的计算
    if (fsize < ssize)
    {
        first = s;
        second = f;
    }
    else
    {
        first = f;
        second = s;
    }

    string last = "";
    
    //重新维护两个字符串长度，防止出现内存空间分配问题
    ssize = second.size();
    fsize = first.size();

    if (first[0] == '-')
    {
        fsize--;
        first = first.substr(1, first.size() - 1);//截取除了负号后面的一段字符串做之后的计算
    }
    if (second[0] == '-')
    {
        ssize--;
        second = second.substr(1, second.size() - 1);//截取除了负号后面的一段字符串做之后的计算
    }

    for (ll i = ssize - 1; i >= 0; i--) //i = 0
    {
        if (i == 0 && i == '-')
        {
            break;
        }
        int carry = 0;
        string sum1 = "";
        for (ll j = fsize - 1; j >= 0; j--)
        {
            int num1 = first[j] - '0';
            int num2 = second[i] - '0';

            if (j == 0 && second[j] == '-')
            {
                break;
            }
            //一旦检测到不是数字的字符就提示用户并且跳转到程序开头使程序重新运行
            if ((num1 > 9 || num1 < 0) || (num2 > 9 || num2 < 0))
            {
                cout << "Wrong input! Please try again!" << endl;
                goto loop;
            }

            int sum = num1 * num2 + carry;//存储两个单位数进行乘法运算的结果

            carry = sum / 10;//存储进位

            int s = sum % 10;//存储和位

            sum1 += s + '0';
        }

        sum1 += carry + '0';//最后不要忘了加上剩余的进位

        //翻转字符串
        string temp = sum1;
        for (ll k = 0; k < sum1.size(); k++)
        {
            temp[k] = sum1[sum1.size() - k - 1];
        }
        sum1 = temp; 

        //last存储上一次进行加法运算的结果
        //初始情况last为空，此时不进行运算，直接将第一次运算的结果赋值给last
        if (last.empty())
        {
            last = sum1;
            res.push(last[sum1.size() - 1]);
            continue;
        }

    
        last = addString(last.substr(0, last.size() - 1), sum1);
        res.push(last[last.size() - 1]);//将个位数先存进结果栈当中，之后便不再考虑这一数位
    }
    for (ll i = last.size() - 2; i >= 0; i--)
    {
        res.push(last[i]);
    }

    string ans = "";
    //将 res 栈顶元素逐一输入到ans字符串当中，并且去掉前导零
    bool flag = false;//标记是否有第一个非零值出现
    while (!res.empty())
    {
        if (res.top() != '0')
        {
            flag = true;
        }
        if (!flag && res.top() == '0')
        {
            res.pop();
            continue;
        }
        ans += res.top();
        res.pop();
    }

    cout << f << " * " << s << " = ";
    //如果结果为负数且答案不为零，就先输出负号
    if (isNegative && ans.size() != 0)
    {
        cout << '-';
    }
    //如果答案为零，则在前面的“去掉前导零”的操作中并不会给字符串添加任何值，即此时字符串为空，需要手动输出'0'
    if (ans.size() == 0)
    {
        cout << "0" << endl;
    }
    //如果答案非零，则输出答案
    else
    {
        cout << ans << endl;
    }

    goto loop;//运算完成之后回到程序起始，再次进行新一轮运算
}

//add two string if both of them are composed of numbers
string addString(string str1, string str2)
{
    string res = "";
    stack<char> result;
    //考虑到字符串的长度可能非常长，长度可能超过所有的基本数据类型的大小，这里采用栈结构存储字符串的每一个字符
    stack<char> stk1;
    stack<char> stk2;
    for (char ch : str1)
    {
        stk1.push(ch);
    }
    for (char ch : str2)
    {
        stk2.push(ch);
    }
    int carry = 0;
    //在两个栈都非空的情况下，分别取两个栈顶元素进行运算，运算规则与前面类似
    while (!stk1.empty() && !stk2.empty())
    {
        //取栈顶元素作为两个加数
        int num1 = stk1.top() - '0';
        int num2 = stk2.top() - '0';
        stk1.pop();
        stk2.pop();

        int sum = num1 + num2 + carry;
        carry = sum / 10;
        result.push(sum % 10 + '0');
    }
    //检测到是哪一个栈非空就直接对该栈的元素进行对结果栈的导入
    if (stk1.empty())
    {
        while (!stk2.empty())
        {
            int num = stk2.top() - '0';

            stk2.pop();
            int sum = carry + num;
            carry = sum / 10;
            result.push(sum % 10 + '0');
        }
    }
    else
    {
        while (!stk1.empty())
        {
            int num = stk1.top() - '0';

            stk1.pop();
            int sum = carry + num;
            carry = sum / 10;
            result.push(sum % 10 + '0');
        }
    }
    //将结果栈中元素导入结果字符串当中
    while (!result.empty())
    {
        res += result.top();
        result.pop();
    }

    return res;
}
