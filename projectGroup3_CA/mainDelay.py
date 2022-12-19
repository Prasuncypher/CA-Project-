import random
import random as rand
import numpy as np
import matplotlib.pyplot as plt


def BTD(binary):  # Converts a given binary string (32 bits) to decimal integer (base 10)
    # 1101
    ch = binary[0]
    binary = binary[::-1]
    # let say the len of binary < 32 bits
    length = len(binary)
    binary = binary + ch * (32 - length)
    num = 0
    for i in range(31):
        if binary[i] == '1':
            num = num + pow(2, i)
    binary = binary[::-1]
    if binary[0] == '1':
        num = num + (pow(2, 31) * -1)
    return num


def reverse(x):
    return x[::-1]


def DTB(num):  # Converts a given integer(base 10) to binary string of 32
    string = ''
    flag = False  # false --> positive number
    if num < 0:
        flag = True  # Negative Number
    num = abs(num)
    while num != 0:
        remainder = num % 2
        string = string + str(remainder)
        num = num // 2
    string = string[::-1]
    length = len(string)
    added = "0" * (32 - length)
    number = added + string
    #    Taking 1's complement of the given binary string
    if flag:
        for i in range(32):
            if number[i] == '1':
                number = number[:i] + '0' + number[i + 1:]
            else:
                number = number[:i] + '1' + number[i + 1:]
        # Adding 1 to the given number
        carry = 0
        # number = number[::-1]
        for i in range(31, -1, -1):
            if number[i] == '1':
                number = number[:i] + '0' + number[i + 1:]
                carry = 1
            else:
                number = number[:i] + '1' + number[i + 1:]
                break
    return number


class Clock:
    def __init__(self, count):
        self.counter = count

    def getCounter(self):
        return self.counter

    def updateCounter(self, val):
        self.counter = self.counter + 1


class CPU:
    def __init__(self):
        self.clock = Clock(0)
        initialValue = 0
        self.program_counter = initialValue  # Initial State
        self.r0 = (0,)  # Immutable Special Register : tuple has been used
        self.registers = list()
        for i in range(31):
            self.registers.append(initialValue)
        self.specialRegisters = dict()
        self.specialRegisters[16384] = initialValue
        self.specialRegisters[16388] = initialValue
        self.specialRegisters[16392] = initialValue
        self.specialRegisters[16396] = initialValue
        self.specialRegisters[16400] = initialValue


class InstructionMemory:
    def __init__(self):  # Constructor for instructionMemory
        self.instructions = list()

    def loadProgram(self, file):  # Loading the complete program in instruction Memory Obj: data member
        # instruction_memory_delay = int(input("Enter the delay(in cycles) for Instruction memory: "))
        for row in file:
            temp = row.strip()
            self.instructions.append(temp)
        return


class DataMemory:
    def __init__(self):  # Constructor
        self.memory = dict()  # Initializing the Dictionary
        self.data_mem_delay = 0

    def initializeMemory(self):  # Initializing Memory with all 0
        initialValue = 0  # initial Value of each of the memory locations
        self.data_memory_delay = int(input("Enter the delay(in cycles) for data memory: "))
        for i in range(1000):  # Number of different memory locations : 1000 (starting from 0)
            addr = i  # address of the memory location
            self.memory[addr] = initialValue


class Fetch:
    def __init__(self):
        self.instruction = ""
        # self.busy = False
        self.FetchinstructionNumber = -1
        pass

    def FetchInstruction(self, instruction):  # Setting the instruction
        # self.busy = True  # Fetch stage is busy till decode stage gets free
        self.instruction = instruction


class Decode:
    def __init__(self):
        # self.busy = False
        self.type = None
        self.DecodeinstructionNumber = -1
        self.result = list()
        pass

    def DecodeInstruction(self, binary):  # decoding the binary instruction fetched from the
        # self.busy = True
        imm = 0
        rs1 = 0
        rs2 = 0
        rd = 0
        binary = reverse(binary)
        if binary[0:7] == "1100110" and binary[12:15] == "000" and binary[25:] == "0000000":
            self.type = "add"
            rd = BTD(reverse(binary[7:12]))
            rs1 = BTD(reverse(binary[15:20]))
            rs2 = BTD(reverse(binary[20:25]))
            self.result = [self.type, rd, rs1, rs2]

        elif binary[0:7] == "1100110" and binary[12:15] == "000" and binary[25:] == "0000010":
            self.type = "sub"
            rd = BTD(reverse(binary[7:12]))
            rs1 = BTD(reverse(binary[15:20]))
            rs2 = BTD(reverse(binary[20:25]))
            self.result = [self.type, rd, rs1, rs2]

        elif binary[0:7] == "1100110" and binary[12:15] == "111" and binary[25:] == "0000000":
            self.type = "and"
            rd = BTD(reverse(binary[7:12]))
            rs1 = BTD(reverse(binary[15:20]))
            rs2 = BTD(reverse(binary[20:25]))
            self.result = [self.type, rd, rs1, rs2]

        elif binary[0:7] == "1100110" and binary[12:15] == "011" and binary[25:] == "0000000":
            self.type = "or"
            rd = BTD(reverse(binary[7:12]))
            rs1 = BTD(reverse(binary[15:20]))
            rs2 = BTD(reverse(binary[20:25]))
            self.result = [self.type, rd, rs1, rs2]

        elif binary[0:7] == "1100110" and binary[12:15] == "100" and binary[25:] == "0000000":
            self.type = "sll"
            rd = BTD(reverse(binary[7:12]))
            rs1 = BTD(reverse(binary[15:20]))
            rs2 = BTD(reverse(binary[20:25]))
            self.result = [self.type, rd, rs1, rs2]

        elif binary[0:7] == "1100110" and binary[12:15] == "101" and binary[25:] == "0000010":
            self.type = "sra"
            rd = BTD(reverse(binary[7:12]))
            rs1 = BTD(reverse(binary[15:20]))
            rs2 = BTD(reverse(binary[20:25]))
            self.result = [self.type, rd, rs1, rs2]

        elif binary[0:7] == "1100100" and binary[12:15] == "000":
            self.type = "addi"
            rd = BTD(reverse(binary[7:12]))
            rs1 = BTD(reverse(binary[15:20]))
            imm = BTD(reverse(binary[20:]))
            self.result = [self.type, rd, rs1, imm]



        elif binary[0:7] == "1100000" and binary[12:15] == "010":  # signals -> rd = m[rs1 + imm]
            self.type = "lw"
            rd = BTD(reverse(binary[7:12]))
            rs1 = BTD(reverse(binary[15:20]))
            imm = BTD(reverse(binary[20:]))
            self.result = [self.type, rd, rs1, imm]

        elif binary[0:7] == "1100011" and binary[12:15] == "000":
            self.type = "beq"
            imm1 = reverse(binary[7:12])
            rs1 = BTD(reverse(binary[15:20]))
            rs2 = BTD(reverse(binary[20:25]))
            imm2 = reverse(binary[25:])
            imm = BTD(imm2 + imm1)
            self.result = [self.type, rs1, rs2, imm]

        elif binary[0:7] == "1100010" and binary[12:15] == "010":
            self.type = "sw"
            imm1 = reverse(binary[7:12])
            rs1 = BTD(reverse(binary[15:20]))
            rs2 = BTD(reverse(binary[20:25]))
            imm2 = reverse(binary[25:])
            imm = BTD(imm2 + imm1)
            self.result = [self.type, rs1, rs2, imm]
        # Remember to unset the flag busy on completing the work , busyFlag : regarding the structural hazard stalling
        elif binary == "11111111111111111111111111111111":
            # print("hi")
            self.type = "storenoc"
            self.result = [self.type, 0, 0, 0]

        elif binary[0:6] == "111111":
            self.type = "loadnoc"
            # print("hi")
            imm1 = reverse(binary[6:15])
            rs1 = BTD(reverse(binary[15:20]))
            rs2 = BTD(reverse(binary[20:25]))
            imm2 = reverse(binary[25:])
            imm = BTD(imm2 + imm1)
            self.result = [self.type, rs2, rs1, imm]

        return self.result


class Xecute:
    def __init__(self):
        # self.busy = False  # Used for checking stalling logics
        self.result = None  # value of register if needs to be udpated
        self.decodeSignals = list()  # storing decode signals for passing it to further stages
        self.XecuteinstructionNumber = -1


    def loadNOC(self, signals, CpuObject):
        # signals = [type, rs2, rs1 , imm]
        # specialRegisters(rs1 + imm) = rs2
        self.decodeSignals = signals
        self.result = CpuObject.registers[signals[1] - 1]

    def storeNOC(self, signals, CpuObject):
        self.decodeSignals = signals
        self.result = 1

    def Add(self, signals, CpuObject, byPassX_X):
        # signals = [type,rd,rs1,rs2] :  we have to add rs1 and rs2 in this function
        val1 = 0
        val2 = 0
        if signals[2] > 0:  # Not register x0
            val1 = CpuObject.registers[signals[2] - 1]
        if signals[3] > 0:  # Not register x0
            val2 = CpuObject.registers[signals[3] - 1]
        # Checking for bypassing values
        # bypass = [rs, value]
        if len(byPassX_X) != 0:
            if signals[2] == byPassX_X[0]:
                val1 = byPassX_X[1]
            elif signals[3] == byPassX_X[0]:
                val2 = byPassX_X[1]

        self.decodeSignals = signals
        self.result = val1 + val2

    def Sub(self, signals, CpuObject, byPassX_X):
        # signals = [type,rd,rs1,rs2] :  we have to add rs1 and rs2 in this function
        # print(signals)
        val1 = 0
        val2 = 0
        if signals[2] > 0:  # Not register x0
            val1 = CpuObject.registers[signals[2] - 1]
        if signals[3] > 0:  # Not register x0
            val2 = CpuObject.registers[signals[3] - 1]
        # Checking for bypassing values
        # print(val1,val2)
        # print(byPassX_X)
        if len(byPassX_X) != 0:
            if signals[2] == byPassX_X[0]:
                val1 = byPassX_X[1]
            elif signals[3] == byPassX_X[0]:
                val2 = byPassX_X[1]
        self.decodeSignals = signals
        # print("sub", CpuObject.registers)
        self.result = val1 - val2
        # print(self.result)

    def AND(self, signals, CpuObject, byPassX_X):
        # signals = [type,rd,rs1,rs2] :  we have to add rs1 and rs2 in this function
        val1 = 0
        val2 = 0
        if signals[2] > 0:  # Not register x0
            val1 = CpuObject.registers[signals[2] - 1]
        if signals[3] > 0:  # Not register x0
            val2 = CpuObject.registers[signals[3] - 1]
        # Checking for bypassing values
        if len(byPassX_X) != 0:
            if signals[2] == byPassX_X[0]:
                val1 = byPassX_X[1]
            elif signals[3] == byPassX_X[0]:
                val2 = byPassX_X[1]
        self.decodeSignals = signals
        self.result = val1 & val2

    def OR(self, signals, CpuObject, byPassX_X):
        # signals = [type,rd,rs1,rs2] :  we have to add rs1 and rs2 in this function
        val1 = 0
        val2 = 0
        if signals[2] > 0:  # Not register x0
            val1 = CpuObject.registers[signals[2] - 1]
        if signals[3] > 0:  # Not register x0
            val2 = CpuObject.registers[signals[3] - 1]
        # Checking for bypassing values
        if len(byPassX_X) != 0:
            if signals[2] == byPassX_X[0]:
                val1 = byPassX_X[1]
            elif signals[3] == byPassX_X[0] :
                val2 = byPassX_X[1]
        self.decodeSignals = signals
        self.result = val1 | val2

    def AddImm(self, signals, CpuObject, byPassX_X):
        # signals = [type,rd,rs1,imm] :  we have to add rs1 and rs2 in this function
        val1 = 0
        if signals[2] > 0:  # Not register x0
            val1 = CpuObject.registers[signals[2] - 1]
        imm = signals[3]
        # Checking for bypassing values
        if len(byPassX_X) != 0 and signals[2] == byPassX_X[0]:
            val1 = byPassX_X[1]
        self.decodeSignals = signals
        self.result = val1 + imm

    def SLL(self, signals, CpuObject, byPassX_X):
        # signals = [type, rd, rs1, rs2]
        val1 = 0
        val2 = 0
        if signals[2] > 0:  # Not register x0
            val1 = CpuObject.registers[signals[2] - 1]
        if signals[3] > 0:  # Not register x0
            val2 = CpuObject.registers[signals[3] - 1]
        # Checking for bypassing values
        if len(byPassX_X) != 0:
            if signals[2] == byPassX_X[0]:
                val1 = byPassX_X[1]
            elif signals[3] == byPassX_X[0]:
                val2 = byPassX_X[1]
        self.decodeSignals = signals
        self.result = val1 << val2

    def SRA(self, signals, CpuObject, byPassX_X):
        # signals = [type, rd, rs1, rs2]
        val1 = 0
        val2 = 0
        if signals[2] > 0:  # Not register x0
            val1 = CpuObject.registers[signals[2] - 1]
        if signals[3] > 0:  # Not register x0
            val2 = CpuObject.registers[signals[3] - 1]
        # Checking for bypassing values
        if len(byPassX_X) != 0:
            if signals[2] == byPassX_X[0]:
                val1 = byPassX_X[1]
            elif signals[3] == byPassX_X[0]:
                val2 = byPassX_X[1]
        self.result = val1 >> val2
        self.decodeSignals = signals

    def LoadWord(self, signals):
        # signals  = [type, rd , rs1 , imm]
        # Nothing has to be done here
        self.result = None
        self.decodeSignals = signals
        # print(signals)

    def StoreWord(self, signals):
        # Nothing has to be done here
        self.result = None
        self.decodeSignals = signals

    def BranchIfEqual(self, signals, CpuObject, byPassX_X):
        # signals = [type, rs1, rs2, imm]
        val1 = 0
        val2 = 0
        self.decodeSignals = signals
        if signals[1] > 0:  # Not register x0
            val1 = CpuObject.registers[signals[1] - 1]
        if signals[2] > 0:  # Not register x0
            val2 = CpuObject.registers[signals[2] - 1]
        # Checking for bypassing values
        if len(byPassX_X) != 0:
            if signals[1] == byPassX_X[0]:
                val1 = byPassX_X[1]
            elif signals[2] == byPassX_X[0]:
                val2 = byPassX_X[1]
        self.result = val1 == val2


class Memory:
    def __init__(self):
        # self.busy = False
        self.mem = False  # False --> no memory operation was there --> write back has to do its work
        self.decodeSignals = list()
        self.MemoryinstructionNumber = -1
        self.result = None

    def storeSignals(self, signals, res):
        self.decodeSignals = signals
        self.result = res
        return

    def loadWord(self, signals, data, CpuObject):
        # lw rd offset(rs1)  val_rd = mem[offset + rs1]
        val = 0  # contains the value of reg : rs1
        if signals[2] > 0:  # Not register x0
            val = CpuObject.registers[signals[2] - 1]
        temp = signals[3]  # offset value (immediate)
        temp = temp + val
        temp = temp  # this is binary address in memory location
        valLoaded = data[temp]  # this is value in binary to be loaded in register
        # CpuObject.registers[signals[1] - 1] = valLoaded
        self.decodeSignals = signals
        self.result = valLoaded
        self.mem = True  # Indication for the next stage

    def storeWord(self, signals, data, CpuObject):
        # signals = [type, rs1 , rs2 , imm] :  M[rs1 + imm] = val(rs2)
        # print("hi", signals)
        valLoaded = 0  # contains the value of reg : rs2
        if signals[2] > 0:  # Not register x0
            valLoaded = CpuObject.registers[signals[2] - 1]
        temp1 = 0  # contains the value of reg : rs1
        if signals[1] > 0:  # Not register x0
            temp1 = CpuObject.registers[signals[1] - 1]
        temp2 = signals[3]  # imm
        temp3 = temp1 + temp2  # Contains the address in binary system
        # temp3 = temp3  # Converting the addr to binary for using dictionary
        # print("mem value: ", temp3)
        data[temp3] = valLoaded  # Updating the memory dictionary
        self.decodeSignals = signals
        self.mem = True  # Indication for the next stage


class WriteBack:
    def __init__(self):
        self.WriteBackinstructionNumber = -1
        # self.busy = False  # used for stalling logic if required
        return

    def writeRegister(self, signals, result, CpuObject):
        # signals = [, rd , ...]
        rd = signals[1]
        if rd == 0:
            return
        # print(result)
        CpuObject.registers[rd - 1] = result
        # print("hi")
        return


def PrintPartialCpuState(cpuObject, writeFile, stallStatus, lis , instructionMem):
    writeFile.write(str('State of Register File at Clock Cycle = ' + str(cpuObject.clock.getCounter()) + ':'))
    writeFile.write("\n")
    output = cpuObject.registers
    writeFile.write(str(output))
    writeFile.write("\n")
    writeFile.write("Printing Special Register File: \n")
    writeFile.write(str(cpuObject.specialRegisters))
    writeFile.write("\n")
    # In case of CPU stalling -> Writing to the file.
    if stallStatus == True:
        writeFile.write("CPU is Stalled!\n")
    # Writing the instruction present in each stage.
    answer = ['Idle']*5
    for i in range(5):
        if lis[i] != -1:
            answer[i] = instructionMem.instructions[lis[i]]
    writeFile.write("Fetch Stage: ")
    writeFile.write(answer[0])
    writeFile.write("\nDecode Stage: ")
    writeFile.write(answer[1])
    writeFile.write("\nXecute Stage: ")
    writeFile.write(answer[2])
    writeFile.write("\nMemory Stage: ")
    writeFile.write(answer[3])
    writeFile.write("\nWriteBack Stage: ")
    writeFile.write(answer[4])
    writeFile.write("\n\n")
    cpuObject.clock.updateCounter(1)
    return


def getTotalInstructionsOfGivenType(instObj, decodeObj):
    ans = [0, 0]
    memoryType = ['lw', 'sw', 'loadnoc', 'storenoc']
    for i in range(len(instObj.instructions)):
        decodeObj.DecodeInstruction(instObj.instructions[i])
        if decodeObj.result[0] in memoryType:
            ans[1] += 1
        else:
            ans[0] += 1
    decodeObj.result = []
    return ans


def instGraph(types):
    r_inst = types[0]
    m_inst = types[1]
    x = np.array(["Register Type Instructions", "Memory Type Instructions"])
    y = np.array([r_inst,m_inst])
    plt.title("Types of Instruction plot.")
    plt.xlabel("Instruction type")
    plt.ylabel("Number of Instructions")
    plt.bar(x,y)
    plt.show()

def DatamemAccessGraph(cycle, location):
    xAxis = np.array(cycle)
    yAxis = np.array(location)
    plt.scatter(xAxis, yAxis)
    plt.title("Data Memory Access Pattern")
    plt.xlabel("Cycle Number")
    plt.ylabel("Memory Address")
    plt.show()

def InstmemAccessGraph(cycle, location):
    xAxis = np.array(cycle)
    yAxis = np.array(location)
    plt.scatter(xAxis, yAxis)
    plt.title("Instruction Memory Access Pattern")
    plt.xlabel("Cycle Number")
    plt.ylabel("Memory Address")
    plt.show()


def stallGraph(cycle):
    x_axis = []
    y_axis = []
    if len(cycle) > 0 :
        temp = [0]*(cycle[len(cycle)-1] + 1)
        for i in cycle:
            temp[i] = 1
        x_axis = list(range(0,cycle[len(cycle)-1]+1))
        y_axis = temp
    plt.scatter(x_axis, y_axis)
    plt.title("Plot displaying the cycles of CPU Stalling.")
    plt.xlabel("Cycle Number: ")
    plt.ylabel("Stalling(1/0)")
    plt.show()


def main():
    #  Opening the input and log Files
    binary = open('testBinary.txt', 'r')
    output = open('logFile.txt', 'w')

    # Instantiating the objects of the System
    cpuObject = CPU()
    instMem = InstructionMemory()
    dataMem = DataMemory()
    fetch = Fetch()
    decode = Decode()
    execute = Xecute()
    memStage = Memory()
    write_back = WriteBack()

    # Loading the program in the instruction memory & data memory
    instMem.loadProgram(binary)
    # print(instMem.instructions)
    dataMem.initializeMemory()

    # Creating a list for storing the number of a given type of instruction.
    # index 0 : register-type instructions & index1: memory-type instructions.
    # Invoking the method to count the result.
    typeOfInstruction = getTotalInstructionsOfGivenType(instMem, decode)

    # Creating a list for storing the instruction-memory access pattern.
    instructionMemoryAccess = []
    instructionCycle = []

    # Creating a list for storing the data-memory access pattern.
    dataMemoryAccess = []
    dataCycle = []

    # Creating a list for storing the cycles against the stalling.
    stallingCycle = []

    # Main Logic of the code
    # while True:
    totalInstructions = 0
    temporary = 4
    # for i in range(20):
    i = -1
    # Declaring stalling flag and delay by the user.
    x = -1 # Stalling for 1 cycle for RAW type (ld R1 -> read R1 in next instruction)
    y = -1
    mem_delay_flag = False
    stalling_flag = False
    branch_flag = False
    new_program_counter = -1
    # Creating a list for pipeline instruction.
    lis = [-1]*5
    while True:
        # check = False  # To check whether current clock cycle is needed or not for the program
        # print(decode.result)
        i = i + 1
        # print(i, memStage.decodeSignals, memStage.result)
        # Step 1 :Write Back Stage
        if mem_delay_flag is False and len(memStage.decodeSignals) != 0:
            # print(i, memStage.decodeSignals, memStage.result)
            # print(memStage.decodeSignals)
            write_back.WriteBackinstructionNumber = memStage.MemoryinstructionNumber
            lis[4] = write_back.WriteBackinstructionNumber
            if memStage.decodeSignals[0] == 'loadnoc':
                cpuObject.specialRegisters[
                    cpuObject.registers[memStage.decodeSignals[2] - 1] + memStage.decodeSignals[3]] = memStage.result

            elif memStage.decodeSignals[0] == 'storenoc':
                cpuObject.specialRegisters[16400] = 1

            elif memStage.decodeSignals[0] != 'sw' and memStage.decodeSignals[
                0] != 'beq':  # Given instruction not a memory instruction
                # print(memStage.decodeSignals, .result , i)
                write_back.writeRegister(memStage.decodeSignals, memStage.result, cpuObject)
                memStage.mem = False
            memStage.decodeSignals = []
        else:
            lis[4] = -1

        # Step 2 : Memory Stage :
        # print(i, execute.decodeSignals, execute.result)
        signals_for_execute = []
        if len(execute.decodeSignals) != 0:
            if mem_delay_flag is True and y > 0 :
                # Case 1 : The current lw/sw instruction have to stall further.
                # y : represents the delay of cycles to be taken further.
                y = y - 1
                # Updating the stalling cycles list.
                stallingCycle.append(i)
                # Writing the log file.
                PrintPartialCpuState(cpuObject, output, True, lis, instMem)
                # Updating the data memory access list.
                dataMemoryAccess.append(dataMemoryAccess[len(dataMemoryAccess) - 1])
                dataCycle.append(i)
                if stalling_flag :
                    x = x - 1
                continue
            elif mem_delay_flag and y == 0:
                # Resetting the variables y and flag to normal conditions.
                y = -1
                mem_delay_flag = False
                # since y=0 -> No further stalling.
                stallingCycle.append(i)
                # Writing the log file.
                PrintPartialCpuState(cpuObject, output, True, lis, instMem)
                # Updating the data memory access list.
                dataMemoryAccess.append(dataMemoryAccess[len(dataMemoryAccess) - 1])
                dataCycle.append(i)
                if stalling_flag :
                    x = x - 1
                continue
            else:
                memStage.MemoryinstructionNumber = execute.XecuteinstructionNumber
                lis[3] = memStage.MemoryinstructionNumber
                signals_for_execute = execute.decodeSignals
                if execute.decodeSignals[0] not in ["sw", 'lw']:
                    # Not a memory operation
                    memStage.storeSignals(execute.decodeSignals, execute.result)
                    # if i == 10:
                    #     print(execute.result)
                else:
                    # Given is a memory operation of type lw or sw.
                    mem_delay_flag = True
                    y = dataMem.data_memory_delay - 1
                    if execute.decodeSignals[0] == "lw":
                        # print(cpuObject.program_counter, execute.decodeSignals)
                        memStage.loadWord(execute.decodeSignals, dataMem.memory, cpuObject)
                        dataMemoryAccess.append(
                            cpuObject.registers[memStage.decodeSignals[2] - 1] + memStage.decodeSignals[3])
                        dataCycle.append(i)
                    elif execute.decodeSignals[0] == "sw":
                        memStage.storeWord(execute.decodeSignals, dataMem.memory, cpuObject)
                        dataMemoryAccess.append(
                            cpuObject.registers[memStage.decodeSignals[1] - 1] + memStage.decodeSignals[3])
                        dataCycle.append(i)
                execute.decodeSignals = []
        else:
            lis[3] = -1

        # Step 3 : Execute Stage
        if len(decode.result) != 0:
            # print(i, decode.result, memStage.decodeSignals)
            temp = decode.result[0]
            # Fetching the by-passed value from the later stages(if required) using the memstage.decodeSignals .
            bypassed_value = []
            X_X_list = ['add', 'sub', 'or', 'and', 'sll', 'sra', 'addi']
            registers_to_be_read = []
            if temp in X_X_list or temp == 'beq':
                # if temp == 'beq':
                #     print("hi")
                registers_to_be_read.append(decode.result[2])
                if temp == 'beq':
                    registers_to_be_read.append(decode.result[1])
                else:
                    registers_to_be_read.append(decode.result[3])
                # print(memStage.decodeSignals)
                if memStage.decodeSignals != [] and memStage.decodeSignals[0] in X_X_list and memStage.decodeSignals[1] in registers_to_be_read:
                    #         By passing is needed from the above instruction pipeline stage.
                    bypassed_value.append(memStage.decodeSignals[1])
                    bypassed_value.append(memStage.result)

            # Working on the stalling because of memory -> execute (RAW) type instruction.
            stalling_RAW_M_X = []
            # print(temp, memStage.decodeSignals)
            if temp in X_X_list or temp == 'beq':
                if stalling_flag is True and x <= 0:
                    stalling_flag = False
                    x = -1
                elif stalling_flag is True and x > 0:
                    # WE HAVE TO STALL FURTHER.
                    stallingCycle.append(i)
                    x = x - 1
                    execute.XecuteinstructionNumber = -1
                    lis[2] = -1
                    PrintPartialCpuState(cpuObject, output, True, lis, instMem)
                    # print("hi")
                    # Updating the Instruction Memory access.
                    if len(instructionCycle) != 0 and instructionCycle[len(instructionCycle)-1] == i-1:
                        instructionMemoryAccess.append(instructionMemoryAccess[len(instructionMemoryAccess)-1])
                        instructionCycle.append(i)
                    continue
                elif stalling_flag is False and len(memStage.decodeSignals) != 0 and memStage.decodeSignals[0] == 'lw':
                    if decode.result[0] == 'beq':
                        registers_to_be_read = [decode.result[1], decode.result[2]]
                        if memStage.decodeSignals[1] in registers_to_be_read:
                            stalling_flag = True
                            # Cycle delaying started (1st cycle counted for current cycle).
                            stallingCycle.append(i)
                            x = dataMem.data_memory_delay
                            execute.XecuteinstructionNumber = -1
                            lis[2] = -1
                            PrintPartialCpuState(cpuObject, output, True, lis, instMem)
                            execute.decodeSignals = []
                            # Updating the Instruction Memory access.
                            if len(instructionCycle) != 0 and instructionCycle[len(instructionCycle) - 1] == i - 1:
                                instructionMemoryAccess.append(
                                    instructionMemoryAccess[len(instructionMemoryAccess) - 1])
                                instructionCycle.append(i)
                            # print("hi")
                            continue
                    elif decode.result[0] in X_X_list:
                        registers_to_be_read = [decode.result[2], decode.result[3]]
                        if memStage.decodeSignals[1] in registers_to_be_read:
                            stalling_flag = True
                            stallingCycle.append(i)
                            # Cycle delaying started (1st cycle counted for current cycle).
                            execute.XecuteinstructionNumber = -1
                            lis[2] = -1
                            x = dataMem.data_memory_delay
                            PrintPartialCpuState(cpuObject, output, True, lis, instMem)
                            execute.decodeSignals = []
                            # Updating the Instruction Memory access.
                            if len(instructionCycle) != 0 and instructionCycle[len(instructionCycle) - 1] == i - 1:
                                instructionMemoryAccess.append(
                                    instructionMemoryAccess[len(instructionMemoryAccess) - 1])
                                instructionCycle.append(i)
                            # print("hi")
                            continue
            execute.XecuteinstructionNumber = decode.DecodeinstructionNumber
            lis[2] = execute.XecuteinstructionNumber
            # print(i, decode.result)
            if temp == "add":
                # if i == 9:
                #     print(bypassed_value)
                execute.Add(decode.result, cpuObject, bypassed_value)
            elif temp == "addi":
                # if i == 4:
                #     print(bypassed_value)
                execute.AddImm(decode.result, cpuObject, bypassed_value)
            elif temp == "sub":
                # if i == 25:
                #     print(bypassed_value)
                execute.Sub(decode.result, cpuObject, bypassed_value)
            elif temp == "and":
                execute.AND(decode.result, cpuObject, bypassed_value)
            elif temp == "or":
                execute.OR(decode.result, cpuObject, bypassed_value)
            elif temp == "lw":
                execute.LoadWord(decode.result)
            elif temp == "sw":
                execute.StoreWord(decode.result)
            elif temp == "sll":
                execute.SLL(decode.result, cpuObject, bypassed_value)
            elif temp == "sra":
                execute.SRA(decode.result, cpuObject, bypassed_value)
            elif temp == "beq":
                # decode = [type, rs1, rs2, imm]
                execute.BranchIfEqual(decode.result, cpuObject, bypassed_value)
                # print(execute.result)
                if execute.result:
                    branch_flag = True
                    effective_offset = decode.result[3] - 2
                    # print("Current Program Counter",BTD(cpuObject.program_counter))
                    new_program_counter = cpuObject.program_counter + effective_offset
                    # print("Updated Program Counter",BTD(cpuObject.program_counter))
                    decode.result = []
                    # execute.decodeSignals = []


            elif temp == "storenoc":
                execute.storeNOC(decode.result, cpuObject)

            elif temp == "loadnoc":
                execute.loadNOC(decode.result, cpuObject)
            decode.result = []
        else:
            lis[2] = -1

        # Step 4 : Decode :
        # print(i, fetch.instruction)
        if len(fetch.instruction) != 0:
            decode.DecodeInstruction(fetch.instruction)
            decode.DecodeinstructionNumber = fetch.FetchinstructionNumber
            lis[1] = decode.DecodeinstructionNumber
            # print(i, decode.result)
            fetch.instruction = ''
        else :
            lis[1] = -1
            # Handling edge case for branch instruction.


        # Step 5 :
        # if not fetch.busy:
        # Fetch Stage is free to work further
        # print(i, totalInstructions, )
        if totalInstructions < len(instMem.instructions):
            totalInstructions = totalInstructions + 1
            # print(i, BTD(cpuObject.program_counter))
            fetch.FetchInstruction(instMem.instructions[cpuObject.program_counter])
            # Updating the lists for instruction memory access.
            instructionMemoryAccess.append(cpuObject.program_counter)
            instructionCycle.append(i)
            fetch.FetchinstructionNumber = cpuObject.program_counter
            lis[0] = fetch.FetchinstructionNumber
            if branch_flag :
                PrintPartialCpuState(cpuObject, output, False, lis, instMem)
                decode.result = []
                fetch.instruction = ''
                # Resetting the flag to False.
                branch_flag = False
                cpuObject.program_counter = new_program_counter
                new_program_counter = -1
                totalInstructions = cpuObject.program_counter
                continue  # Move to a new cycle
            else:
                cpuObject.program_counter = cpuObject.program_counter + 1  # updating the program counter by 1
        else:
            lis[0] = -1
            if branch_flag :
                PrintPartialCpuState(cpuObject, output, False, lis, instMem)
                decode.result = []
                fetch.instruction = ''
                # Resetting the flag to False.
                branch_flag = False
                cpuObject.program_counter = new_program_counter
                new_program_counter = -1
                totalInstructions = cpuObject.program_counter
                continue  # Move to a new cycle
        # if not check:  # Checking whether some work was done or not
        #     break
        # print("3\n"),
        PrintPartialCpuState(cpuObject, output, False, lis, instMem)
        if fetch.instruction == '':
            temporary = temporary - 1
            if temporary == 0:
                break
    # Closing the text files opened
    binary.close()
    output.write("\nEnd State of Memory\n")
    output.write(str(dataMem.memory))
    output.close()
    # Plotting the graphs
    # print(typeOfInstruction)
    instGraph(typeOfInstruction)
    DatamemAccessGraph(dataCycle, dataMemoryAccess)
    InstmemAccessGraph(instructionCycle, instructionMemoryAccess)
    stallGraph(stallingCycle)
    return


if __name__ == '__main__':
    # Invoking the main method.
    main()

'''
2) Memory State of CPU ==> Graphs
3) User Input(x:delay) 
'''


