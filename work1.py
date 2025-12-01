class Work1:
    def __init__(self, code=None, name=None, weap=None):
        self.heroes = {}
        if code is not None:
            self.heroes[str(code)] = {"name": name, "weap": weap}

    def check_hero(self, code):
        code = str(code)
        if code in self.heroes:
            info = self.heroes[code]
            print(f'英雄编号：{code}，英雄姓名：{info["name"]}，英雄武器：{info["weap"]}')
        else:
            print('英雄编号不存在')
        
    def add_hero(self, code, name, weap):
        code = str(code)
        if code in self.heroes:
            print('英雄编号已存在')
            return
        self.heroes[code] = {"name": name, "weap": weap}
        print('添加成功')

    def del_hero(self, code):
        code = str(code)
        if code in self.heroes:
            del self.heroes[code]
            print('删除成功')
        else:
            print('英雄编号不存在')

    def mod_hero(self, code, name, weap):
        code = str(code)
        if code in self.heroes:
            self.heroes[code]["name"] = name
            self.heroes[code]["weap"] = weap
            print('修改成功')
        else:
            print('英雄编号不存在')

    def menu(self):
        print('******英雄信息管理系统******')
        print('1、查看英雄信息（请输入英雄编号）')
        print('2、添加英雄到英雄信息管理系统')
        print('3、修改英雄')
        print('4、从英雄信息管理系统删除英雄')
        print('5、退出')


    def run(self):
        while True:
            self.menu()
            choice = int(input('请选择功能：'))
            if choice == '1':
                code = int(input('请输入英雄编号：'))
                self.check_hero(code)
            elif choice == '2':
                code = int(input('请输入英雄编号：'))
                name = input('请输入英雄姓名：')
                weap = input('请输入英雄武器：')
                self.add_hero(code, name, weap)
            elif choice == '3':
                code = int(input('请输入要修改的英雄编号：'))
                name = input('请输入新的英雄姓名：')
                weap = input('请输入新的英雄武器：')
                self.mod_hero(code, name, weap)
            elif choice == '4':
                code = int(input('请输入要删除的英雄编号：'))
                self.del_hero(code)
            elif choice == '5':
                print('已退出')
                break
            else:
                print('无效选择，请重新输入')

if __name__ == '__main__':
    app = Work1()
    app.run()
