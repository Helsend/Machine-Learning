import jieba.posseg as psg
#我们使用jieba只对文中的中文进行分词处理。并以段为分割，将每段的分词结果存为一个列表，而数据列表则包括多个列表。即我们的数据集。 
def loadDataSet():
    santi_text = open('D:\\mywork\\jiqixuexi\\fp_growth\\three_body.txt',encoding='UTF-8').read()
    text = santi_text.split('\n')     #以段为分割
    while '' in text:       #去除列表中的空格
            text.remove('')
    dataSet=[]
    for list in text:
        santiWord = [x.word for x in psg.cut(list) if len(x.word)>=2]    #对每个段落的分词结果存到列表中
        dataSet.append(santiWord)
    return dataSet

#在有了数据集列表后，我们就可以使用FP-Growth算法对数据进行处理。首先将我们的数据列表转换为字典类型，以数据为键，值为出现次数
"""
函数说明：从列表到字典的类型转换函数
parameters：
    dataSet -数据集列表
return：
    retDict -数据集字典
"""
def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1   #将列表项转换为forzenset类型并作为字典的键值，值为该项的出现次数
    return retDict

"""
类说明：FP树数据结构
function：
    __init__ -初始化节点
        nameValue -节点值
        numOccur -节点出现次数
        parentNode -父节点
    inc -对count变量增加给定值
    disp -将树以文本形式显示
"""
#构建树的数据结果，树中节点包括名字、计数值、相似元素链表、父节点、孩子节点
class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue   #存放节点名字
        self.count = numOccur   #节点计数值
        self.nodeLink = None    #链接相似的元素值
        self.parent = parentNode    #当前节点的父节点
        self.children = {}  #空字典变量，存放节点的子节点
   
    def inc(self, numOccur):
        self.count += numOccur
   
    def disp(self, ind=1): #ind为节点的深度
        print(' '*ind, self.name, ' ', self.count)
        for child in self.children.values():
            child.disp(ind+1)   #递归遍历树的每个节点

"""
函数说明：FP树构建函数
parameters:
    dataSet -字典型数据集
    minSup -最小支持度
return：
    retTree -FP树
    headerTable -头指针表
"""
#基于数据集定义FP树构建函数，遍历两次数据集，第一次产生头指针表，第二次更新树结果
def createTree(dataSet, minSup = 1):
    headerTable = {}    #创建空字典，存放头指针
    for trans in dataSet: #遍历数据集
        for item in trans:  #遍历每个元素项
            headerTable[item] = headerTable.get(item,0)+dataSet[trans]  #以节点为key，节点的次数为值
    tmpHeaderTab = headerTable.copy()
    for k in tmpHeaderTab.keys():    #遍历头指针表
        if headerTable[k] < minSup:     #如果出现次数小于最小支持度
            del(headerTable[k])     #删掉该元素项
    freqItemSet = set(headerTable.keys())   #将字典的键值保存为频繁项集合
    if len(freqItemSet) == 0: return None, None #如果过滤后的频繁项为空，则直接返回
    for k in headerTable:
        headerTable[k] = [headerTable[k], None] #使用nodeLink
        #print(headerTable)
    retTree = treeNode('Null Set',1,None) #创建树的根节点
    for tranSet, count in dataSet.items():    #再次遍历数据集
        localD = {} #创建空字典
        for item in tranSet:
            if item in freqItemSet: #该项是频繁项
                localD[item] = headerTable[item][0] #存储该项的出现次数，项为键值
        if len(localD) > 0:
            orderedItems = [v[0] for v in sorted(localD.items(),key=lambda p: p[1],reverse = True)] #基于元素项的绝对出现频率进行排序
            #print(orderedItems)
            updateTree(orderedItems, retTree, headerTable, count)   #使用orderedItems更新树结构
    return retTree, headerTable
#将路径上的节点添加到树中，每次添加都需更新当前节点的孩子节点以及头指针表
"""
函数说明：FP树生长函数
parameters:
    items -项集
    inTree -树节点
    headerTable -头指针表
    count -项集出现次数
return：
    None
"""
def updateTree(items, inTree, headerTable, count):
    if items[0] in inTree.children:     #首先测试items的第一个元素项是否作为子节点存在
        inTree.children[items[0]].inc(count)  #如果存在，则更新该元素项的计数
    else:
        inTree.children[items[0]] = treeNode(items[0],count,inTree) #如果不存在，创建一个新的treeNode并将其作为子节点添加到树中
        if headerTable[items[0]][1] == None:    #将该项存到头指针表中的nodelink
            headerTable[items[0]][1] = inTree.children[items[0]]    #记录nodelink
        else:
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]]) #若已经存在nodelink，则更新至链表尾
    if len(items) > 1:
        updateTree(items[1::],inTree.children[items[0]],headerTable,count)  #迭代，每次调用时会去掉列表中的第一个元素
#更新头指针表函数，记录当前节点的位置到该节点类型相似元素的链表尾
"""
函数说明：确保节点链接指向树中该元素项的每一个实例
parameters:
    nodeToTest -需要更新的头指针节点
    targetNode -要指向的实例
return：
    None
"""
def updateHeader(nodeToTest, targetNode):
    while(nodeToTest.nodeLink!=None):   #从头指针表的nodelink开始，直到达到链表末尾
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode    #记录当前元素项的实例

#建立条件基时，对每个节点进行路径的回溯，直至到达树根节点。
"""
函数说明：上溯FP树
parameters：
    leafNode -节点
    prefixPath -该节点的前缀路径
return:
    None
"""

def ascendTree(leafNode, prefixPath):
    if leafNode.parent != None: #如果该节点的父节点存在
        prefixPath.append(leafNode.name)    #将其加入到前缀路径中
        ascendTree(leafNode.parent,prefixPath)  #迭代调用自身上溯
#在遍历头指针表时，对该节点的相似元素链表进行遍历，对链表中的每个元素调用回溯函数
"""
函数说明：遍历某个元素项的nodelink链表
parameters：
    basePat -头指针表中元素
    treeNode -该元素项的nodelist链表节点
return:
    condPats -该元素项的条件模式基
"""
def findPrefixPath(basePat, treeNode):
    condPats = {} #创建空字典，存放条件模式基
    while treeNode != None:
        prefixPath = []
        ascendTree(treeNode, prefixPath)    #寻找该路径下实例的前缀路径
        if len(prefixPath)>1:   #如果有前缀路径
            condPats[frozenset(prefixPath[1:])] = treeNode.count #记录该路径的出现次数，出现次数为该路径下起始元素项的计数值
        treeNode = treeNode.nodeLink
    return condPats
#根据当前的频繁项集，从头指针表中的单元素集开始，基于每个元素寻找频繁项集，一直到树中没有元素为止。
"""
函数说明：在FP树中寻找频繁项
parameters:
    inTree -FP树
    headerTable -当前元素前缀路径的头指针表
    minSup -最小支持度
    preFix -当前元素的前缀路径
    freqItemList -频繁项集
return:
    None
"""
def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: p[1][0])] #按照出现次数从小到大排序
    for basePat in bigL:  #从最少次数的元素开始
        newFreqSet = preFix.copy()  #复制前缀路径
        newFreqSet.add(basePat)     #将当前元素加入路径
        freqItemList.append(newFreqSet)     #将该项集加入频繁项集
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])    #找到当前元素的条件模式基
        myCondTree, myHead = createTree(condPattBases, minSup)  #过滤低于阈值的item，基于条件模式基建立FP树
        if myHead != None: #如果FP树中存在元素项 
            # 递归的挖掘每个条件FP树，累加后缀频繁项集       
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)  #递归调用自身函数，直至FP树中没有元素
if __name__ == '__main__':
    parsedDat = loadDataSet()
    initSet = createInitSet(parsedDat)
    myFPtree,myHeaderTab = createTree(initSet,700)
    myFreqList = []
    mineTree(myFPtree,myHeaderTab,300,set([]),myFreqList)
    print("出现过300次及以上的频繁项个数：\n",len(myFreqList))
    print("出现过300次及以上的频繁项：\n",myFreqList)