<template>
  <v-app :style="{ background: $vuetify.theme.themes.light.background }">
    <v-layout row>
      <v-flex md8>
        <v-app :style="{ background: $vuetify.theme.themes.dark.background }" class="rounded-tr-xl rounded-br-xl">
          <v-container>
            <v-row>
              <v-col cols="12" sm="12">
                <v-app-bar color="rgba(0,0,0,0)" flat class="mx-8 mb-8 mt-3">
                  <v-select :items="cameras" v-model="selected_camera" class="choice-camera mr-3" label="Choice camera"
                    dense>
                  </v-select>
                  <v-btn color="primary" class="mr-3 " elevation="2" @click="Choice_Cam_Video()">Request</v-btn>
                  <span class="camera-name">You choose: {{ selected_camera }}</span>
                  <v-spacer></v-spacer>
                  
                </v-app-bar>
              </v-col>
              <v-col cols="12" sm="12">
                <v-list two-line> </v-list>
              </v-col>
              <v-col cols="12" sm="24">
                <v-card class="mx-12 rounded-tl-xl rounded-tr-xl rounded-bl-xl rounded-br-xl mt-n15">
                  <v-list-item three-line id="video-analysis">
                    <img  :src="'http://127.0.0.1:8000/' + imgPreUrl +'/'" class="w-100 h-100" alt="">
                  </v-list-item>
                </v-card>
              </v-col>
              <v-col cols="12" sm="12" class="mb-3">
                <h5 data-v-6a4aa1a0="" class="h3 text-black mb-0" id="title-of-chart" style="margin: 30px;border-bottom: 1px solid;"> 
                  Tổng lượng xe theo các giờ trong ngày 
                </h5>
                <canvas id="myChart" width="400" height="200"></canvas>
              </v-col>
              <v-col cols="12" sm="12" >

                <canvas id="myChartLine" style="display:none" width="400" height="200"></canvas>
              </v-col>
            </v-row>
          </v-container>
        </v-app>
      </v-flex>
      <v-flex md4>
        <v-app :style="{ background: $vuetify.theme.themes.light.background }">
          <v-container>
            <v-row>
              <v-col cols="12" md="3">
                <v-row>
                  <v-col cols="12" md="8">
                    <v-list two-line subheader class="ml-n8" style="background-color: inherit">
                      <v-list-item class="v-boder-left v-border-blue">
                        <v-list-item-content>
                          <v-list-item-subtitle>Car</v-list-item-subtitle>
                          <v-list-item-title>{{ task.car }}</v-list-item-title>
                        </v-list-item-content>
                      </v-list-item>
                    </v-list>
                  </v-col>
                </v-row>
              </v-col>
              <v-col cols="12" md="3">
                <v-row>
                  <v-col cols="12" md="10">
                    <v-list two-line subheader class="ml-n8">
                      <v-list-item class="v-boder-left v-border-yellow">
                        <v-list-item-content>
                          <v-list-item-subtitle>Truck</v-list-item-subtitle>
                          <v-list-item-title>{{ task.truck }}</v-list-item-title>
                        </v-list-item-content>
                      </v-list-item>
                    </v-list>
                  </v-col>
                </v-row>
              </v-col>
              <v-col cols="12" md="3">
                <v-row>
                  <v-col cols="12" md="10">
                    <v-list two-line subheader class="ml-n8">
                      <v-list-item class="v-boder-left v-border-red">
                        <v-list-item-content>
                          <v-list-item-subtitle>Motobike</v-list-item-subtitle>
                          <v-list-item-title>{{ task.motorcycle }}</v-list-item-title>
                        </v-list-item-content>
                      </v-list-item>
                    </v-list>
                  </v-col>
                </v-row>
              </v-col>
              <v-col cols="12" md="3">
                <v-row>
                  <v-col cols="12" md="10">
                    <v-list two-line subheader class="ml-n8">
                      <v-list-item class="v-boder-left v-border-pink">
                        <v-list-item-content>
                          <v-list-item-subtitle>Bus</v-list-item-subtitle>
                          <v-list-item-title>{{ task.bus }}</v-list-item-title>
                        </v-list-item-content>
                      </v-list-item>
                    </v-list>
                  </v-col>
                </v-row>
              </v-col>
              <v-col cols="12" sm="12">
                <v-list>
                  <v-list-item class="d-flex justify-content-between pl-0 pr-0">
                    <v-item-content style="width: 30%" v-for="item in list_time_analysis_choice" :key="item.id"
                      class="item-analysis">
                      <v-btn class="mr-3" style="width: 100%" :class="item.active ? 'btn-analysis-active' : ''"
                        @click="choiceTimeHandle(item.id)">{{ item.name }}</v-btn>
                    </v-item-content>
                  </v-list-item>
                </v-list>
              </v-col>
              <v-col cols="12" sm="12" class="mb-3" :class="{
                'd-flex': list_time_analysis_choice[0].active == true,
              }" v-show="list_time_analysis_choice[0].active == true">
                <v-card class="mr-12 rounded-tl-xl rounded-tr-xl rounded-bl-xl rounded-br-xl border-color-primary" flat>
                  <date-picker v-model="selected_date" :config="optionsCalendar" class="text-center"
                    @change="onChange()"></date-picker>
                </v-card>
                <v-btn class="ml-3" style="background-color: #00897b; color: #fff" @click="requestDay">Request</v-btn>
              </v-col>
              <v-col v-show="list_time_analysis_choice[1].active == true" cols="12" sm="12" class="mb-3">
                <select class="select-choice select-month text-center mr-4" style="width: 40%" v-model="selected_month">
                  <option v-for="month in months" :key="month.id" :value="month" :selected="selected_month === month">
                    {{ month.name }}
                  </option>
                </select>
                <select v-model="selected_year" class="select-choice select-year text-center" style="width: 20%;padding: 10px;border: 2px solid #00897b;border-radius: 10px;"
                  @change="onChangeYear()">
                  <option v-for="(year, index) in years" :key="index">
                    {{ year }}
                  </option>
                </select>
                <v-btn class="ml-3" style="background-color: #00897b; color: #fff" @click="requestMonth">Request</v-btn>
              </v-col>
              <v-col v-show="list_time_analysis_choice[2].active == true" cols="12" sm="12" class="mb-3">
                <select v-model="selected_year" class="select-choice select-year text-center" style="width: 30%;padding: 10px;border: 2px solid #00897b;border-radius: 10px;">
                  <option v-for="(year, index) in years" :key="index">
                    {{ year }}
                  </option>
                </select>
                <v-btn class="ml-3" style="background-color: #00897b; color: #fff" @click="requestYear">Request</v-btn>
              </v-col>

              <v-col cols="12" sm="12">
                <h5 data-v-6a4aa1a0="" class="h3 text-black mb-0" style="text-align: center;"> Phần trăm loại xe </h5>
              </v-col>
              <v-col cols="12" sm="12">
                <canvas id="myChartDougnut" width="400" height="400"></canvas>
              </v-col>
            </v-row>
          </v-container>
        </v-app>
      </v-flex>
    </v-layout>
    <!-- test -->
  </v-app>
</template>
<script src="https://cdn.jsdelivr.net/npm/chart.js@3.8.0/dist/chart.min.js"></script>
<script type="text/javascript" src="https://yourweb.com/inc/chart.utils.js"></script>
<script src="https://momentjs.com/downloads/moment.min.js"></script>
<script>

import axios from 'axios';
import { GChart } from "vue-google-charts/legacy";
export default {
  name: "home",
  components: {
  GChart,
  },
  type: 'bar',
  d: new Date(),
  async created(){
    // await this.fetchAPI_Year();
    await this.fetchAPI();
  },
  data: () => ({
    BE : [],
    myChartLine:"",
    duyen: new Date(2016, 9,  16),
    myChart:"",
    myChartDougnut: "",
    day: (new Date()).getDate(),
    month: (new Date()).getMonth() + 1,
    year: (new Date()).getFullYear(),
    cam: 'Cam1',
    // option for date-pick
    optionsCalendar: {
      format: "DD/MM/YYYY",
      useCurrent: true,
    },
    imgPreUrl : "webcam1",
    task: [],
    width: 2,
    radius: 10,
    padding: 8,
    lineCap: "round",
    value: [0, 2, 5, 9, 5, 10, 3, 5, 0, 0, 1, 8, 2, 9, 0],
    fill: false,
    type: "trend",
    autoLineWidth: false,
    date_current_choice: new Date(),
    options: {
      format: "DD/MM/YYYY",
      useCurrent: true,
      scales: {
        y: {
          beginAtZero: true
        }
      },
    },
    list_time_analysis_choice: [
      {
        id: 0,
        name: "Day",
        active: true,
      },
      {
        id: 1,
        name: "Month",
        active: false,
      },
      {
        id: 2,
        name: "Year",
        active: false,
      },
    ],
    chartData: {
      "2017-05-13": 2,
      "2017-05-14": 3,
      "2017-05-15": 4,
    },
    months: [
      {
        id: 0,
        number: 1,
        name: "January",
      },
      {
        id: 1,
        number: 2,
        name: "February",
      },
      {
        id: 2,
        number: 3,
        name: "March",
      },
      {
        id: 3,
        number: 4,
        name: "April",
      },
      {
        id: 4,
        number: 5,
        name: "May",
      },
      {
        id: 5,
        number: 6,
        name: "June",
      },
      {
        id: 6,
        number: 7,
        name: "June",
      },
      {
        id: 7,
        number: 8,
        name: "August",
      },
      {
        id: 8,
        number: 9,
        name: "September",
      },
      {
        id: 9,
        number: 10,
        name: "October",
      },
      {
        id: 10,
        number: 11,
        name: "November",
      },
      {
        id: 11,
        number: 12,
        name: "December",
      },
    ],
    days: {
        "1": 0,
        "2": 0,
        "3": 0,
        "4": 0,
        "5": 0,
        "6": 0,
        "7": 0,
        "8": 0,
        "9": 0,
        "10": 0,
        "11": 0,
        "12": 0,
        "13": 0,
        "14": 0,
        "15": 0,
        "16": 0,
        "17": 0,
        "18": 0,
        "19": 0,
        "20": 0,
        "21": 20,
        "22": 30,
        "23": 50,
        "24": 30,
        "25": 40,
        "26": 70,
        "27": 50,
        "28": 110,
        "29": 0,
        "30": 0
    },
    selected_month: {
        id: 0,
        number: 1,
        name: "January",
      },
    selected_date: new Date(),
    years: ["2020", "2021", "2022"],
    selected_year: "2022",
    cameras: ["Cam1", "Cam2"],
    selected_camera: "Cam1",
  }),
  computed: {
    theme() {
      return this.$vuetify.theme.dark ? "dark" : "light";
    }
  },
  methods: {
    loadDataToChart(){
      if (this.task.chart_title == "Hour"){
        this.fetchAPI();
        this.myChartLine.data.datasets[0].data = Object.values(this.task.hour);
        this.myChartLine.data.datasets[0].label = "HOURS ON A DAY"
        this.myChartLine.data.labels = Object.keys(this.task.hour);
        this.myChartLine.update();
        document.getElementById('myChart').style.display = "none";
        document.getElementById('myChartLine').style.display = "block";
        document.getElementById('title-of-chart').innerHTML = "Tổng lượng xe theo các giờ trong ngày";
      }
      else if (this.task.chart_title == "Day"){
        this.fetchAPI();
        this.myChartLine.data.datasets[0].data = Object.values(this.task.chart);
        this.myChartLine.data.datasets[0].label = "DAYS IN MONTH"
        this.myChartLine.data.labels = Object.keys(this.task.chart);
        this.myChartLine.update();
        document.getElementById('myChart').style.display = "none";
        document.getElementById('myChartLine').style.display = "block";
        document.getElementById('title-of-chart').innerHTML="Tổng lượng xe theo các ngày trong tháng";
      }
      else {
        this.fetchAPI();
        this.myChart.data.datasets[0].data = Object.values(this.task.chart);
        this.myChart.data.datasets[0].label = "MONTHS IN YEAR"
        this.myChart.data.labels = Object.keys(this.task.chart);
        this.myChart.data.datasets[0].backgroundColor = [
                  'rgba(255, 0, 0, 0.2)',
                  'rgba(255, 127, 0, 0.2)',
                  'rgba(255, 255, 0, 0.2)',
                  'rgba(127, 255, 0, 0.2)',
                  'rgba(0, 255, 0, 0.2)',
                  'rgba(0, 255, 127, 0.2)',
                  'rgba(0, 255, 255, 0.2)',
                  'rgba(0, 127, 255, 0.2)',
                  'rgba(0, 0, 255, 0.2)',
                  'rgba(127, 0, 255, 0.2)',
                  'rgba(255, 0, 255, 0.2)',
                  'rgba(255, 0, 127, 0.2)',
          ];
        this.myChart.data.datasets[0].borderColor= [
                  'rgba(255, 0, 0, 1)',
                  'rgba(255, 127, 0, 1)',
                  'rgba(255, 255, 0, 1)',
                  'rgba(127, 255, 0, 1)',
                  'rgba(0, 255, 0, 1)',
                  'rgba(0, 255, 127, 1)',
                  'rgba(0, 255, 255, 1)',
                  'rgba(0, 127, 255, 1)',
                  'rgba(0, 0, 255, 1)',
                  'rgba(127, 0, 255, 1)',
                  'rgba(255, 0, 255, 1)',
                  'rgba(255, 0, 127, 1)',
          ];
        this.myChart.borderWidth = 1;
        this.myChart.update();
        document.getElementById('myChartLine').style.display = "none";
        document.getElementById('myChart').style.display = "block";
        document.getElementById('title-of-chart').innerHTML="Tổng lượng xe theo các tháng trong năm";
      }
      this.set_data_doughnut()
    },
    onChange() {
      alert(this.date);
    },
    set_data_doughnut(){
      let sum_vehicle = this.task.truck + this.task.bus + this.task.car + this.task.motorcycle
      this.myChartDougnut.data.datasets[0].data =[parseFloat(this.task.truck/sum_vehicle*100).toFixed(2),parseFloat(this.task.bus/sum_vehicle*100).toFixed(2),parseFloat(this.task.car/sum_vehicle*100).toFixed(2),parseFloat(this.task.motorcycle/sum_vehicle*100).toFixed(2)]
      this.myChartDougnut.update()
    },
    choiceTimeHandle(id) {
      this.list_time_analysis_choice
        .filter((x) => x.id === id)
        .map((x) => (x.active = true));
      this.list_time_analysis_choice
        .filter((x) => x.id !== id)
        .map((x) => (x.active = false));
    },
    async fetchAPI() {
      console.log("fetchAPI");
      const data_fetch = this.task.hour;
      console.log(
        "Data: ",
        this.task
      );
    },
    async fetchAPI_Year() {
      console.log("fetchAPI_YEAR");
      const data_fetch = this.task.chart;
      console.log("t",this.myChart.data);
      console.log(
        "Data: ",
        this.task
      );
      console.log("data_fetch: ", data_fetch);
      this.myChart.data.datasets[0].data = data_fetch;
      this.myChart.update();
    },
    async get_Tasks() {
      await axios({
        method: 'get',
        url: "http://localhost:8000/vehicle?year=" + this.selected_year + "&day=" + "&month=" + this.selected_month.number + "&camera=" + this.selected_camera,
        // url: "http://127.0.0.1:8000/vehicle?year=2022&day=&month=&camera=Cam1"
      }).then(response => this.task = response.data);
      this.loadDataToChart()
    },
    async requestMonth() {
      try {

        console.log(this.months.filter(function(el){return el.name == "November"})[0])
        axios
          .get(
            `http://localhost:8000/vehicle?year=${this.selected_year}&day=&month=${this.selected_month.number}&camera=${this.selected_camera}`
          )
          .then((response) => {
            this.task = response.data;
            this.loadDataToChart();
          });
      } catch (e) {
        console.log("Error: ", e);
      }
    },
    async requestDay() {
      try {
        let dateString = this.selected_date.toString().split("/");
        let date = dateString[0];
        let month = dateString[1];
        let year = dateString[2];
        axios
          .get(
            `http://localhost:8000/vehicle?year=${year}&day=${date}&month=${month}&camera=${this.selected_camera}`
          )
          .then((response) => {
            this.task = response.data;
            this.loadDataToChart();
          });
      } catch (e) {
        console.log("Error: ", e);
      }
    },
    async requestYear() {
      try {
        console.log("@@@: " + this.selected_year);
        axios
          .get(
            `http://localhost:8000/vehicle?year=${this.selected_year}&day=&month=&camera=${this.selected_camera}`
          )
          .then((response) => {
            this.task = response.data;
            console.log(
              "@@@@: " + this.selected_year
            );
            this.loadDataToChart();
          });
      } catch (e) {
        console.log("Error: ", e);
      }
    },
    getTime() {
      alert(this.selected_month)
      alert(this.selected_year)
      axios({
        method: 'get',
        // url: "http://localhost:8000/vehicle?year=" + this.selected_year + "&day=" + "&month=" + this.selected_month.number + "&camera=" + this.cam,
        url: "http://127.0.0.1:8000/vehicle?year=2022&day=21&month=4&camera=Cam1"

      }).then(response => this.task = response.data);
      console.log(this.task.chart_title)

      console.log('task: ', this.task)
    },
    Choice_Cam_Video(){
      console.log("cam1")
      if(this.selected_camera == "Cam1"){
        this.imgPreUrl = "webcam1"
        console.log("cam1")
      }
      else{
        this.imgPreUrl = "webcam2"
        console.log("cam2")
      }
      axios({
        method: 'get',
        url: "http://localhost:8000/vehicle?year=" + this.selected_year + "&day=" + "&month=" + "&camera=" + this.selected_camera,
        // url: "http://127.0.0.1:8000/vehicle?year=2022&day=&month=&camera=Cam1"
      }).then(response => this.task = response.data);
      this.loadDataToChart()
    }
  },
  mounted() {   
    const data_mychart = {
      labels: Object.keys(this.months),
      datasets: [{
        label: ['# of Votes'],
        data: [12, 19, 3, 5, 10, 3,6,8,16,6,12,6],
        backgroundColor: [
          'rgba(255, 0, 0, 0.2)',
          'rgba(255, 127, 0, 0.2)',
          'rgba(255, 255, 0, 0.2)',
          'rgba(127, 255, 0, 0.2)',
          'rgba(0, 255, 0, 0.2)',
          'rgba(0, 255, 127, 0.2)',
          'rgba(0, 255, 255, 0.2)',
          'rgba(0, 127, 255, 0.2)',
          'rgba(0, 0, 255, 0.2)',
          'rgba(127, 0, 255, 0.2)',
          'rgba(255, 0, 255, 0.2)',
          'rgba(255, 0, 127, 0.2)',
        ],
        borderColor: [
          'rgba(255, 0, 0, 1)',
          'rgba(255, 127, 0, 1)',
          'rgba(255, 255, 0, 1)',
          'rgba(127, 255, 0, 1)',
          'rgba(0, 255, 0, 1)',
          'rgba(0, 255, 127, 1)',
          'rgba(0, 255, 255, 1)',
          'rgba(0, 127, 255, 1)',
          'rgba(0, 0, 255, 1)',
          'rgba(127, 0, 255, 1)',
          'rgba(255, 0, 255, 1)',
          'rgba(255, 0, 127, 1)',
        ],
        borderWidth: 1
      }]
    };
    const data_lines = {
      labels: Object.keys(this.days),
      datasets: [{
        label: 'My First Dataset',
        data: Object.values(this.days),
        fill: false,
        borderColor: 'rgb(75, 192, 192)',
        tension: 0.1
      }]
    }
    const data_doughnut = {
      labels: [
        'Truck (%)',
        'Bus (%)',
        'Car (%)',
        'Motorbike (%)'
      ],
      datasets: [{
        label: 'Phần trăm loại xe (%)',
        data: [300, 50, 100,45],
        backgroundColor: [
          'rgb(154, 156, 233)',
          'rgb(162, 185, 237)',
          'rgb(163, 220, 239)',
          'rgb(174, 238, 224)'
        ],
        hoverOffset: 4
      }]
    };
    const ctx = document.getElementById('myChart').getContext('2d');
    this.myChart = new Chart(ctx, {
      type: 'bar',
      data: data_mychart, 
      options: {
        scales: {
          y: {
            beginAtZero: true
          }
        }
      }
    });
    const ctxDougnut = document.getElementById('myChartDougnut').getContext('2d');
    this.myChartDougnut = new Chart(ctxDougnut, {
        type: 'doughnut',
        data:data_doughnut,
    });
    const ctxLine = document.getElementById('myChartLine').getContext('2d');
    this.myChartLine = new Chart(ctxLine, {
        type: 'line',
        data:data_lines,
    });
    this.get_Tasks();
  }
};
</script>
<style lang="scss" scoped>
date-picker {
  input {
    padding: 10px;
    text-align: center;
  }
}

.v-boder-left {
  border-left: 3px solid #00897b;
}

.v-boder-left.v-border-red {
  border-left: 3px solid red;
}

.v-boder-left.v-border-yellow {
  border-left: 3px solid rgb(251, 251, 170);
}

.v-boder-left.v-border-pink {
  border-left: 3px solid rgb(241, 184, 194);
  ;
}

.v-boder-left.v-border-blue {
  border-left: 3px solid rgb(166, 166, 255);
  ;
}

.item-analysis {
  button {
    background-color: #fff !important;
    outline: 1px solid #00897b;
  }

  button:hover {
    background-color: #00897b !important;
    color: #fff !important;
  }
  select.select-year{
    padding: 10px;
    border: 2px solid #00897b;
    border-radius: 10px;
  }
  .btn-analysis-active {
    background-color: #00897b !important;
    color: #fff !important;
  }
}

#video-analysis {
  overflow: hidden;
  height: 500px;
  padding: 0px;
}

.calendar-date {
  border: 1px solid #00897b;
}

.select-month {
  padding: 10px;
  border: 2px solid #00897b;
  border-radius: 10px;
}

.select-month:focus-visible {
  border: 2px solid #00897b;
}
</style>