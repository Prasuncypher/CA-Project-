import random
import random as rand


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
        initialValue = "0" * 32
        self.program_counter = initialValue  # Initial State
        self.r0 = (initialValue,)  # Immutable Special Register : tuple has been used
        self.registers = list()
        for i in range(31):
            self.registers.append(initialValue)


class InstructionMemory:
    def __init__(self):  # Constructor for instructionMemory
        self.instructions = list()

    def loadProgram(self, file):  # Loading the complete program in instruction Memory Obj: data member
        for row in file:
            temp = row.strip()
            self.instructions.append(temp)
        return


class DataMemory:
    def __init__(self):  # Constructor
        self.memory = dict()  # Initializing the Dictionary

    def initializeMemory(self):  # Initializing Memory with all 0
        initialValue = "0" * 32  # initial Value of each of the memory locations
        for i in range(1000):  # Number of different memory locations : 1000 (starting from 0)
            addr = DTB(i)  # address of the memory location
            self.memory[addr] = initialValue


class Fetch:
    def __init__(self):
        self.instruction = ""
        # self.busy = False
        pass

    def FetchInstruction(self, instruction):  # Setting the instruction
        # self.busy = True  # Fetch stage is busy till decode stage gets free
        self.instruction = instruction


class Decode:
    def __init__(self):
        # self.busy = False
        self.type = None
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

        elif binary[0:7] == "1100000" and binary[12:15] == "010":
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
        return self.result


class Xecute:
    def __init__(self):
        # self.busy = False  # Used for checking stalling logics
        self.result = None  # value of register if needs to be udpated
        self.decodeSignals = list()  # storing decode signals for passing it to further stages

    def Add(self, signals, CpuObject):
        # signals = [type,rd,rs1,rs2] :  we have to add rs1 and rs2 in this function
        val1 = 0
        val2 = 0
        if signals[2] > 0:  # Not register x0
            val1 = BTD(CpuObject.registers[signals[2] - 1])
        if signals[3] > 0:  # Not register x0
            val2 = BTD(CpuObject.registers[signals[3] - 1])
        self.decodeSignals = signals
        self.result = val1 + val2

    def Sub(self, signals, CpuObject):
        # signals = [type,rd,rs1,rs2] :  we have to add rs1 and rs2 in this function
        val1 = 0
        val2 = 0
        if signals[2] > 0:  # Not register x0
            val1 = BTD(CpuObject.registers[signals[2] - 1])
        if signals[3] > 0:  # Not register x0
            val2 = BTD(CpuObject.registers[signals[3] - 1])
        self.decodeSignals = signals
        self.result = val1 - val2

    def AND(self, signals, CpuObject):
        # signals = [type,rd,rs1,rs2] :  we have to add rs1 and rs2 in this function
        val1 = 0
        val2 = 0
        if signals[2] > 0:  # Not register x0
            val1 = BTD(CpuObject.registers[signals[2] - 1])
        if signals[3] > 0:  # Not register x0
            val2 = BTD(CpuObject.registers[signals[3] - 1])
        self.decodeSignals = signals
        self.result = val1 and val2

    def OR(self, signals, CpuObject):
        # signals = [type,rd,rs1,rs2] :  we have to add rs1 and rs2 in this function
        val1 = 0
        val2 = 0
        if signals[2] > 0:  # Not register x0
            val1 = BTD(CpuObject.registers[signals[2] - 1])
        if signals[3] > 0:  # Not register x0
            val2 = BTD(CpuObject.registers[signals[3] - 1])
        self.decodeSignals = signals
        self.result = val1 or val2

    def AddImm(self, signals, CpuObject):
        # signals = [type,rd,rs1,imm] :  we have to add rs1 and rs2 in this function
        val1 = 0
        if signals[2] > 0:  # Not register x0
            val1 = BTD(CpuObject.registers[signals[2] - 1])
        imm = signals[3]
        self.decodeSignals = signals
        self.result = val1 + imm

    def SLL(self, signals, CpuObject):
        # signals = [type, rd, rs1, rs2]
        val1 = 0
        val2 = 0
        if signals[2] > 0:  # Not register x0
            val1 = BTD(CpuObject.registers[signals[2] - 1])
        if signals[3] > 0:  # Not register x0
            val2 = BTD(CpuObject.registers[signals[3] - 1])
        self.decodeSignals = signals
        self.result = val1 << val2

    def SRA(self, signals, CpuObject):
        # signals = [type, rd, rs1, rs2]
        val1 = 0
        val2 = 0
        if signals[2] > 0:  # Not register x0
            val1 = BTD(CpuObject.registers[signals[2] - 1])
        if signals[3] > 0:  # Not register x0
            val2 = BTD(CpuObject.registers[signals[3] - 1])
        self.result = val1 >> val2
        self.decodeSignals = signals

    def LoadWord(self, signals):
        # Nothing has to be done here
        self.result = None
        self.decodeSignals = signals

    def StoreWord(self, signals):
        # Nothing has to be done here
        self.result = None
        self.decodeSignals = signals

    def BranchIfEqual(self, signals, CpuObject):
        # signals = [type, rs1, rs2, imm]
        val1 = 0
        val2 = 0
        self.decodeSignals = signals
        if signals[1] > 0:  # Not register x0
            val1 = BTD(CpuObject.registers[signals[1] - 1])
        if signals[2] > 0:  # Not register x0
            val2 = BTD(CpuObject.registers[signals[2] - 1])
        self.result = val1 == val2


class Memory:
    def __init__(self):
        # self.busy = False
        self.mem = False  # False --> no memory operation was there --> write back has to do its work
        self.decodeSignals = list()
        self.result = None

    def storeSignals(self, signals, res):
        self.decodeSignals = signals
        self.result = res
        return

    def loadWord(self, signals, data, CpuObject):
        # lw rd offset(rs1)  val_rd = mem[offset + rs1]
        val = 0  # contains the value of reg : rs1
        if signals[2] > 0:  # Not register x0
            val = BTD(CpuObject.registers[signals[2] - 1])
        temp = signals[3]  # offset value (immediate)
        temp = temp + val
        temp = DTB(temp)  # this is binary address in memory location
        valLoaded = data[temp]  # this is value in binary to be loaded in register
        # CpuObject.registers[signals[1] - 1] = valLoaded
        self.decodeSignals = signals
        self.result = BTD(valLoaded)
        self.mem = True  # Indication for the next stage

    def storeWord(self, signals, data, CpuObject):
        # signals = [type, rs1 , rs2 , imm] :  M[rs1 + imm] = val(rs2)
        valLoaded = 0  # contains the value of reg : rs2
        if signals[2] > 0:  # Not register x0
            valLoaded = BTD(CpuObject.registers[signals[2] - 1])
        temp1 = 0  # contains the value of reg : rs1
        if signals[1] > 0:  # Not register x0
            temp1 = BTD(CpuObject.registers[signals[1] - 1])
        temp2 = signals[3]  # imm
        temp3 = temp1 + temp2  # Contains the address in decimal system
        temp3 = DTB(temp3)  # Converting the addr to binary for using dictionary
        # print("mem value: ", temp3)
        data[temp3] = DTB(valLoaded)  # Updating the memory dictionary
        self.decodeSignals = signals
        self.mem = True  # Indication for the next stage


class WriteBack:
    def __init__(self):
        # self.busy = False  # used for stalling logic if required
        return

    def writeRegister(self, signals, result, CpuObject):
        # signals = [, rd , ...]
        rd = signals[1]
        if rd == 0:
            return
        CpuObject.registers[rd - 1] = DTB(result)
        return


def PrintPartialCpuState(cpuObject, writeFile):
    string = 'State of Register File at Clock Cycle =', cpuObject.clock.getCounter(), ':'
    writeFile.write(str(string))
    writeFile.write("\n")
    output = cpuObject.registers
    writeFile.write(str(output))
    writeFile.write("\n")
    cpuObject.clock.updateCounter(1)
    return


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
    # print(instMem.instructions[BTD(cpuObject.program_counter)])
    # return
    # Main Logic of the code
    # while True:
    totalInstructions = 0
    for i in range(27):
        # check = False  # To check whether current clock cycle is needed or not for the program
        # Step 1 :Write Back Stage
        # print(decode.result)
        # print(i, decode.result)
        if len(decode.result) != 0 and len(memStage.decodeSignals) != 0:
            # print(i, memStage.decodeSignals)
            # print(memStage.decodeSignals)
            if memStage.decodeSignals[0]!='sw' and memStage.decodeSignals[0]!='beq':  # Given instruction not a memory instruction
                # print(memStage.decodeSignals, .result , i)
                write_back.writeRegister(memStage.decodeSignals, memStage.result, cpuObject)
                memStage.mem = False
            memStage.decodeSignals = []

        # Step 2 : Memory Stage :
        if len(execute.decodeSignals) != 0:
            if execute.decodeSignals[0] not in ["sw", 'lw']:
                # Not a memory operation
                memStage.storeSignals(execute.decodeSignals, execute.result)
            else:
                # Given is a memory operation
                if execute.decodeSignals[0] == "lw":
                    memStage.loadWord(execute.decodeSignals, dataMem.memory, cpuObject)
                elif execute.decodeSignals[0] == "sw":
                    memStage.storeWord(execute.decodeSignals, dataMem.memory, cpuObject)
            execute.decodeSignals = []

        # Step 3 : Execute Stage
        # print(i, decode.result)
        if len(decode.result) != 0 :
            # print(i, decode.result)
            if len(memStage.decodeSignals)!=0 and memStage.decodeSignals[0] == 'lw':
                registerToBeWritten = memStage.decodeSignals[1]
                # print(i, memStage.decodeSignals , decode.result)
                lis = ['lw', 'sw', 'beq', 'addi']
                if decode.result[0] not in lis:
                    # Decode instruction have all registers rd,rs1,rs2
                    if registerToBeWritten in decode.result:  # RAW and WAW
                        PrintPartialCpuState(cpuObject, output)
                        execute.decodeSignals = []
                        continue
                else :
                    # Other instructions have only 2 registers
                    if registerToBeWritten == decode.result[1] or registerToBeWritten == decode.result[2]:
                        # print("hi")
                        execute.decodeSignals = []
                        PrintPartialCpuState(cpuObject, output)
                        continue
            elif len(memStage.decodeSignals)!=0 and memStage.decodeSignals[0] == 'sw':
                # we have to check only for WAR case
                registerToBeWritten = decode.result[1]
                lis = ['lw', 'sw', 'beq', 'addi']
                if decode.result[0] not in lis:
                    # Decode instruction have all registers rd,rs1,rs2
                    if registerToBeWritten in decode.result:  # RAW and WAW
                        PrintPartialCpuState(cpuObject, output)
                        execute.decodeSignals = []
                        continue
                elif decode.result[0] == 'addi':
                    if registerToBeWritten == memStage.decodeSignals[1] or registerToBeWritten == memStage.decodeSignals[2]:
                        PrintPartialCpuState(cpuObject, output)
                        execute.decodeSignals = []
                        continue
            temp = decode.result[0]
            # print(i, temp)
            if temp == "add":
                execute.Add(decode.result, cpuObject)
            elif temp == "addi":
                execute.AddImm(decode.result, cpuObject)
            elif temp == "sub":
                execute.Sub(decode.result, cpuObject)
            elif temp == "and":
                execute.AND(decode.result, cpuObject)
            elif temp == "or":
                execute.OR(decode.result, cpuObject)
            elif temp == "lw":
                execute.LoadWord(decode.result)
            elif temp == "sw":
                execute.StoreWord(decode.result)
            elif temp == "sll":
                execute.SLL(decode.result, cpuObject)
            elif temp == "sra":
                execute.SRA(decode.result, cpuObject)
            elif temp == "beq":
                # print(i, "hi")
                execute.BranchIfEqual(decode.result, cpuObject)
                # print(execute.result)
                if execute.result:
                    effective_offset = decode.result[3] - 2
                    # print("Current Program Counter",BTD(cpuObject.program_counter))
                    cpuObject.program_counter= DTB(BTD(cpuObject.program_counter) + effective_offset)
                    # print("Updated Program Counter",BTD(cpuObject.program_counter))
                    fetch.instruction = ""
                    totalInstructions = BTD(cpuObject.program_counter)
                    decode.result = []
                    execute.decodeSignals = []
                    memStage.decodeSignals = []
                    PrintPartialCpuState(cpuObject, output)
                    continue  # Move to a new cycle

        # Step 4 : Decode :
        if len(fetch.instruction) != 0:
            decode.DecodeInstruction(fetch.instruction)

        # Step 5 :
        # if not fetch.busy:
        # Fetch Stage is free to work further
        # print(i, totalInstructions, )
        if totalInstructions < len(instMem.instructions):
            totalInstructions = totalInstructions + 1
            # print(i, BTD(cpuObject.program_counter))
            fetch.FetchInstruction(instMem.instructions[BTD(cpuObject.program_counter)])
            cpuObject.program_counter = DTB(BTD(cpuObject.program_counter) + 1)  # updating the program counter by 1

        # if not check:  # Checking whether some work was done or not
        #     break
        # print("3\n")
        PrintPartialCpuState(cpuObject, output)
    # Closing the text files opened
    binary.close()
    # print("End State of Memory\n", dataMem.memory)
    output.close()
    return


if __name__ == '__main__':
    main()
