<template>
  <div class="main">
    <h1>This is about main</h1>
    <line-chart :data="chartData"></line-chart>
    <pie-chart
      :data="[
        ['Nokia', 180],
        ['Samsung', 120],
      ]"
    ></pie-chart>
    <line-chart download="boom"></line-chart>
    <line-chart :data="{ '2017-05-13': 2, '2017-05-14': 5 }"></line-chart>
    <pie-chart
      :data="[
        ['Car', 44],
        ['Motobike', 23],
        ['Truck', 18],
      ]"
    ></pie-chart>
    <column-chart
      :data="[
        ['Sun', 32],
        ['Mon', 46],
        ['Tue', 28],
        ['Web', 21],
        ['Thu', 20],
        ['Fri', 13],
        ['Sat', 27],
      ]"
    ></column-chart>
    <v-col cols="12" sm="12">
      <v-card
        class="mx-12 rounded-tl-xl rounded-tr-xl rounded-bl-xl rounded-br-xl mt-n10"
      >
        <v-app-bar color="rgba(0,0,0,0)" flat class="ma-8">
          <v-spacer></v-spacer>
          <v-btn color="teal" text
            >2019
            <v-icon right> mdi-chevron-down </v-icon>
          </v-btn>
          <v-btn color="teal" rounded dark depressed>Year</v-btn>
          <v-btn text>Mounth</v-btn>
        </v-app-bar>
        <template>
          <v-sparkline
            :value="value"
            color="teal"
            :smooth="radius || false"
            :padding="padding"
            :line-width="width"
            :stroke-linecap="lineCap"
            :fill="fill"
            :type="type"
            :auto-line-width="autoLineWidth"
            auto-draw
          ></v-sparkline>
        </template>
      </v-card>
    </v-col>
  </div>
</template>
<script>
export default {
  name: "about",
  data() {
    return {
      width: 2,
      radius: 10,
      padding: 8,
      lineCap: "round",
      value: [0, 2, 5, 9, 5, 10, 3, 5, 0, 0, 1, 8, 2, 9, 0],
      fill: false,
      type: "trend",
      autoLineWidth: false,
      chartData: {
        "2017-05-13": 2,
        "2017-05-14": 3,
        "2017-05-15": 4,
      },
    };
  },
};
</script>

<style lang="scss" scoped>
.about {
  .about-video {
    width: 800px;
    height: 500px;
    background-color: red;
    iframe {
      width: 100%;
      height: 100%;
      background-color: pink;
    }
  }
}
</style>
